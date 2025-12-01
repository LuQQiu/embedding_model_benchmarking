/*
 * ONNX Runtime C++ Embedding Server
 *
 * High-performance embedding inference server using ONNX Runtime C++ API
 * Features:
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

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

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

// Global state
struct ServerState {
    unique_ptr<Ort::Env> env;
    unique_ptr<Ort::Session> session;
    unique_ptr<Ort::SessionOptions> session_options;
    unique_ptr<tokenizers::Tokenizer> tokenizer;

    string model_name;
    string model_path;
    int max_seq_length;
    int embedding_dim;

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
    state.model_path = model_config["paths"]["onnx"].as<string>();

    cout << "Model: " << model_name << endl;
    cout << "Max sequence length: " << state.max_seq_length << endl;
    cout << "Embedding dimension: " << state.embedding_dim << endl;
    cout << "Model path: " << state.model_path << endl;
}

// Initialize ONNX Runtime
void init_onnx() {
    auto start = chrono::high_resolution_clock::now();

    // Create environment
    state.env = make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnx_cpp_server");

    // Create session options
    state.session_options = make_unique<Ort::SessionOptions>();
    state.session_options->SetIntraOpNumThreads(thread::hardware_concurrency());
    state.session_options->SetInterOpNumThreads(thread::hardware_concurrency());
    state.session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Create session
    state.session = make_unique<Ort::Session>(*state.env, state.model_path.c_str(), *state.session_options);

    cout << "✓ ONNX session created" << endl;

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
        {"framework", "onnx-cpp"},
        {"model_name", state.model_name},
        {"model_configuration", {
            {"max_seq_length", state.max_seq_length},
            {"embedding_dim", state.embedding_dim}
        }},
        {"model_load_time_ms", state.model_load_time_ms},
        {"total_requests", state.total_requests.load()},
        {"runtime_version", std::to_string(ORT_API_VERSION).c_str()},
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

        // Create ONNX tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );

        vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_len)};

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids_vec.data(), input_ids_vec.size(),
            input_shape.data(), input_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask_vec.data(), attention_mask_vec.size(),
            input_shape.data(), input_shape.size()
        );

        // Run inference
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"token_embeddings"};

        vector<Ort::Value> input_tensors;
        input_tensors.push_back(move(input_ids_tensor));
        input_tensors.push_back(move(attention_mask_tensor));

        auto output_tensors = state.session->Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), 2,
            output_names, 1
        );

        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

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
        {"service", "ONNX Runtime C++ Embedding Server"},
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
    cout << "ONNX Runtime C++ Server - Starting" << endl;
    cout << "======================================================================" << endl;

    state.start_time = chrono::steady_clock::now();

    try {
        // Load configuration
        load_config();

        // Initialize ONNX Runtime
        init_onnx();

        cout << "  ONNX Runtime version: " << std::to_string(ORT_API_VERSION).c_str() << endl;
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
