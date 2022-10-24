#include "model.h"


namespace jetson {

Model::Model(YAML::Node& config) : config_(config) {
}

int Model::Init() {}

Model::~Model() {
}

}  // namespace jetson