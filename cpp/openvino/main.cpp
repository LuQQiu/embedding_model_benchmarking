/*
 * OpenVINO C++ Embedding Server
 *
 * High-performance embedding inference server using OpenVINO C++ API
 * Features Intel-optimized inference for maximum CPU performance
 * - HTTP REST API using cpp-httplib
 * - FastTokenizer for text tokenization
 * - Mean pooling and L2 normalization
 * - JSON request/response handling
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <queue>
#include <mutex>
#include <condition_variable>

// OpenVINO
#include <openvino/openvino.hpp>

// HTTP server
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

// JSON parsing
#include "nlohmann/json.hpp"

// YAML parsing
#include "yaml-cpp/yaml.h"

// Tokenizer
#include "tokenizers_cpp.h"

using json = nlohmann::json;
using namespace std;

// Inference request pool for thread-safe concurrent inference
class InferRequestPool {
private:
    vector<ov::InferRequest> pool;
    queue<size_t> available;
    mutex mtx;
    condition_variable cv;

public:
    InferRequestPool(ov::CompiledModel& compiled_model, size_t pool_size) {
        for (size_t i = 0; i < pool_size; i++) {
            pool.push_back(compiled_model.create_infer_request());
            available.push(i);
        }
    }

    // Acquire an inference request from the pool
    pair<ov::InferRequest&, size_t> acquire() {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [this] { return !available.empty(); });
        size_t idx = available.front();
        available.pop();
        return {pool[idx], idx};
    }

    // Release an inference request back to the pool
    void release(size_t idx) {
        lock_guard<mutex> lock(mtx);
        available.push(idx);
        cv.notify_one();
    }
};

// Global state
struct ServerState {
    ov::Core core;
    shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    unique_ptr<InferRequestPool> infer_pool;
    unique_ptr<tokenizers::Tokenizer> tokenizer;

    string model_name;
    string model_path;
    int max_seq_length;
    int embedding_dim;
    size_t pool_size;

    double model_load_time_ms;
    atomic<uint64_t> total_requests{0};

    chrono::steady_clock::time_point start_time;
};

ServerState state;

// Helper: L2 normalization
void normalize_embeddings(vector<float>& embeddings, size_t batch_size, size_t embedding_dim) {
    for (size_t i = 0; i < batch_size; i++) {
        float norm = 0.0f;
        size_t offset = i * embedding_dim;

        // Calculate L2 norm
        for (size_t j = 0; j < embedding_dim; j++) {
            float val = embeddings[offset + j];
            norm += val * val;
        }
        norm = sqrt(norm);
        norm = max(norm, 1e-9f);

        // Normalize
        for (size_t j = 0; j < embedding_dim; j++) {
            embeddings[offset + j] /= norm;
        }
    }
}

// Helper: Mean pooling
vector<float> mean_pooling(
    const vector<float>& last_hidden_state,
    const vector<int64_t>& attention_mask,
    size_t batch_size,
    size_t seq_length,
    size_t embedding_dim
) {
    vector<float> pooled(batch_size * embedding_dim, 0.0f);

    for (size_t b = 0; b < batch_size; b++) {
        vector<float> sum_embeddings(embedding_dim, 0.0f);
        float sum_mask = 0.0f;

        for (size_t s = 0; s < seq_length; s++) {
            int64_t mask_val = attention_mask[b * seq_length + s];
            sum_mask += static_cast<float>(mask_val);

            for (size_t e = 0; e < embedding_dim; e++) {
                size_t idx = b * seq_length * embedding_dim + s * embedding_dim + e;
                sum_embeddings[e] += last_hidden_state[idx] * mask_val;
            }
        }

        sum_mask = max(sum_mask, 1e-9f);

        for (size_t e = 0; e < embedding_dim; e++) {
            pooled[b * embedding_dim + e] = sum_embeddings[e] / sum_mask;
        }
    }

    return pooled;
}

// Load configuration
void load_config() {
    string model_name = getenv("MODEL_NAME") ? getenv("MODEL_NAME") : "embeddinggemma-300m";
    state.model_name = model_name;

    // Load models.yaml
    YAML::Node config = YAML::LoadFile("../config/models.yaml");
    YAML::Node models = config["models"];

    if (!models[model_name]) {
        throw runtime_error("Model " + model_name + " not found in config");
    }

    YAML::Node model_config = models[model_name];
    state.max_seq_length = model_config["max_seq_length"].as<int>();
    state.embedding_dim = model_config["embedding_dim"].as<int>();
    state.model_path = model_config["paths"]["openvino"].as<string>() + "/model.xml";

    cout << "Model: " << model_name << endl;
    cout << "Max sequence length: " << state.max_seq_length << endl;
    cout << "Embedding dimension: " << state.embedding_dim << endl;
    cout << "Model path: " << state.model_path << endl;
}

// Initialize OpenVINO
void init_openvino() {
    auto start = chrono::high_resolution_clock::now();

    // Read model
    state.model = state.core.read_model(state.model_path);

    // Compile model for CPU
    state.compiled_model = state.core.compile_model(state.model, "CPU");

    // Create inference request pool (size = CPU count for optimal parallelism)
    state.pool_size = min(static_cast<size_t>(thread::hardware_concurrency()), static_cast<size_t>(64));
    state.infer_pool = make_unique<InferRequestPool>(state.compiled_model, state.pool_size);

    cout << "✓ OpenVINO model compiled (inference pool size: " << state.pool_size << ")" << endl;

    // Load tokenizer
    string tokenizer_path = state.model_path;
    size_t last_slash = tokenizer_path.find_last_of('/');
    if (last_slash != string::npos) {
        tokenizer_path = tokenizer_path.substr(0, last_slash) + "/tokenizer.json";
    }
    // Load tokenizer
    ifstream tokenizer_file(tokenizer_path);
    string tokenizer_json((istreambuf_iterator<char>(tokenizer_file)),
                          istreambuf_iterator<char>());
    state.tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json);


    cout << "✓ Tokenizer loaded" << endl;

    auto end = chrono::high_resolution_clock::now();
    state.model_load_time_ms = chrono::duration<double, milli>(end - start).count();

    cout << "✓ Model loaded in " << state.model_load_time_ms << "ms" << endl;
}

// Health endpoint
void handle_health(const httplib::Request& req, httplib::Response& res) {
    json response = {
        {"status", "healthy"},
        {"model_loaded", true}
    };
    res.set_content(response.dump(), "application/json");
}

// Info endpoint
void handle_info(const httplib::Request& req, httplib::Response& res) {
    auto uptime = chrono::duration_cast<chrono::seconds>(
        chrono::steady_clock::now() - state.start_time
    ).count();

    json response = {
        {"framework", "openvino-cpp"},
        {"model_name", state.model_name},
        {"model_configuration", {
            {"max_seq_length", state.max_seq_length},
            {"embedding_dim", state.embedding_dim}
        }},
        {"model_load_time_ms", state.model_load_time_ms},
        {"total_requests", state.total_requests.load()},
        {"runtime_version", ov::get_openvino_version().buildNumber},
        {"device", "CPU"},
        {"cpu_count", thread::hardware_concurrency()},
        {"uptime_seconds", uptime}
    };
    res.set_content(response.dump(), "application/json");
}

// Embed endpoint
void handle_embed(const httplib::Request& req, httplib::Response& res) {
    auto start = chrono::high_resolution_clock::now();

    try {
        // Parse request
        json request_json = json::parse(req.body);
        if (!request_json.contains("texts") || !request_json["texts"].is_array()) {
            res.status = 400;
            res.set_content("{\"error\": \"Missing or invalid 'texts' field\"}", "application/json");
            return;
        }

        vector<string> texts = request_json["texts"].get<vector<string>>();
        if (texts.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"No texts provided\"}", "application/json");
            return;
        }

        size_t batch_size = texts.size();

        // Tokenize
        vector<vector<int32_t>> token_ids_batch;
        for (const auto& text : texts) {
            token_ids_batch.push_back(state.tokenizer->Encode(text));
        }

        // Find max length in batch
        size_t max_len = 0;
        for (const auto& ids : token_ids_batch) {
            max_len = max(max_len, ids.size());
        }
        max_len = min(max_len, static_cast<size_t>(state.max_seq_length));

        // Prepare input tensors
        vector<int64_t> input_ids_vec;
        vector<int64_t> attention_mask_vec;
        input_ids_vec.reserve(batch_size * max_len);
        attention_mask_vec.reserve(batch_size * max_len);

        for (const auto& ids : token_ids_batch) {

            for (size_t i = 0; i < max_len; i++) {
                if (i < ids.size()) {
                    input_ids_vec.push_back(static_cast<int64_t>(ids[i]));
                    attention_mask_vec.push_back(1);
                } else {
                    input_ids_vec.push_back(0);
                    attention_mask_vec.push_back(0);
                }
            }
        }

        // Acquire an inference request from the pool
        auto [infer_request, pool_idx] = state.infer_pool->acquire();

        // Create OpenVINO tensors
        ov::Shape input_shape = {batch_size, max_len};
        ov::Tensor input_ids_tensor(ov::element::i64, input_shape, input_ids_vec.data());
        ov::Tensor attention_mask_tensor(ov::element::i64, input_shape, attention_mask_vec.data());

        // Set input tensors
        infer_request.set_tensor("input_ids", input_ids_tensor);
        infer_request.set_tensor("attention_mask", attention_mask_tensor);

        // Run inference
        infer_request.infer();

        // Get output
        ov::Tensor output_tensor = infer_request.get_tensor("token_embeddings");
        float* output_data = output_tensor.data<float>();
        ov::Shape output_shape = output_tensor.get_shape();

        size_t output_batch = output_shape[0];
        size_t output_seq_len = output_shape[1];
        size_t output_embed_dim = output_shape[2];

        // Copy to vector
        size_t total_elements = output_batch * output_seq_len * output_embed_dim;
        vector<float> last_hidden_state(output_data, output_data + total_elements);

        // Mean pooling
        vector<float> embeddings = mean_pooling(
            last_hidden_state, attention_mask_vec,
            batch_size, max_len, output_embed_dim
        );

        // L2 normalization
        normalize_embeddings(embeddings, batch_size, output_embed_dim);

        // Release inference request back to pool
        state.infer_pool->release(pool_idx);

        auto end = chrono::high_resolution_clock::now();
        double inference_time_ms = chrono::duration<double, milli>(end - start).count();

        // Update counter
        state.total_requests++;

        // Build response
        json response;
        response["embeddings"] = json::array();
        for (size_t i = 0; i < batch_size; i++) {
            vector<float> embedding(
                embeddings.begin() + i * output_embed_dim,
                embeddings.begin() + (i + 1) * output_embed_dim
            );
            response["embeddings"].push_back(embedding);
        }
        response["inference_time_ms"] = inference_time_ms;

        res.set_content(response.dump(), "application/json");

    } catch (const exception& e) {
        cerr << "Error in embedding: " << e.what() << endl;
        res.status = 500;
        json error = {{"error", e.what()}};
        res.set_content(error.dump(), "application/json");
    }
}

// Root endpoint
void handle_root(const httplib::Request& req, httplib::Response& res) {
    json response = {
        {"service", "OpenVINO C++ Embedding Server"},
        {"status", "running"},
        {"endpoints", {
            {"health", "/health"},
            {"info", "/info"},
            {"embed", "/embed (POST)"}
        }}
    };
    res.set_content(response.dump(), "application/json");
}

int main() {
    cout << "======================================================================" << endl;
    cout << "OpenVINO C++ Server - Starting" << endl;
    cout << "======================================================================" << endl;

    state.start_time = chrono::steady_clock::now();

    try {
        // Load configuration
        load_config();

        // Initialize OpenVINO
        init_openvino();

        cout << "  OpenVINO version: " << ov::get_openvino_version().buildNumber << endl;
        cout << "  CPU count: " << thread::hardware_concurrency() << endl;
        cout << endl;
        cout << "Server ready on http://0.0.0.0:8000" << endl;
        cout << "======================================================================" << endl;

        // Create HTTP server
        httplib::Server svr;

        // Register endpoints
        svr.Get("/", handle_root);
        svr.Get("/health", handle_health);
        svr.Get("/info", handle_info);
        svr.Post("/embed", handle_embed);

        // Start server
        svr.listen("0.0.0.0", 8000);

    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
