#pragma once
#include "common.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

class SceneNode {
public:
	SceneNode(const uint id) : id(id) {};
	
	uint getId() const {return id; }

private:
	uint id;
};

class SceneInstance: Node {
public:

	
private:
	transform3f transform;
	uint child;
};

class SceneGroup : public Node {
public:
	
	
private:
	std::vector<SceneInstance> children;
};

KRR_NAMESPACE_END