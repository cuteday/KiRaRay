#include "scenegraph.h"
#include "window.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

const string& SceneGraphLeaf::getName() const { 
	auto node = getNode();
	if (node) return node->getName();
	static const string emptyString = "";
	return emptyString;
}

void SceneGraphLeaf::setName(const std::string &name) const {
	auto node = getNode();
	if (node) node->setName(name);
	else Log(Error, "Set name %s for a leaf without a node", name.c_str());
}

SceneGraphLeaf::SharedPtr MeshInstance::clone() {
	return std::make_shared<MeshInstance>(mMesh);
}

SceneGraphLeaf::SharedPtr SceneAnimation::clone() {
	auto copy = std::make_shared<SceneAnimation>();
	for (const auto &channel : mChannels) 
		copy->addChannel(std::make_shared<SceneAnimationChannel>(
			channel->getSampler(), channel->getTargetNode(), channel->getAttribute()));
	return std::static_pointer_cast<SceneGraphLeaf>(copy);
}

SceneGraphLeaf::SharedPtr PointLight::clone() {
	return std::make_shared<PointLight>(color, scale);
}

SceneGraphLeaf::SharedPtr DirectionalLight::clone() {
	return std::make_shared<PointLight>(color, scale);
}

SceneGraphLeaf::SharedPtr InfiniteLight::clone() {
	auto light = std::make_shared<InfiniteLight>(color, scale);
	light->setTexture(texture);
	return light;
}

SceneGraphLeaf::SharedPtr HomogeneousVolume::clone() {
	return std::make_shared<HomogeneousVolume>(sigma_t, albedo, g, Le);
}

SceneGraphLeaf::SharedPtr VDBVolume::clone() {
	return std::make_shared<VDBVolume>(sigma_t, albedo, g, densityGrid, temperatureGrid, albedoGrid,
									   scale, LeScale, temperatureScale, temperatureOffset);
}

void SceneGraphNode::setTransform(const Vector3f *translation, const Quaternionf *rotation,
								  const Vector3f *scaling) {
	if (scaling) mScaling = *scaling;
	if (rotation) mRotation = *rotation;
	if (translation) mTranslation = *translation;

	mHasLocalTransform = true;
	mUpdateFlags |= UpdateFlags::LocalTransform;
	propagateUpdateFlags(UpdateFlags::SubgraphTransform);
}

void SceneGraphNode::setScaling(const Vector3f &scaling) {
	setTransform(nullptr, nullptr, &scaling);
}

void SceneGraphNode::setRotation(const Quaternionf &rotation) {
	setTransform(nullptr, &rotation, nullptr);
}

void SceneGraphNode::setTranslation(const Vector3f &translation) {
	setTransform(&translation, nullptr, nullptr);
}

void SceneGraphNode::setLeaf(const SceneGraphLeaf::SharedPtr &leaf) {
	auto graph = mGraph.lock();

	if (mLeaf) {
		mLeaf->mNode.reset();
		if (graph) graph->unregisterLeaf(mLeaf);
	}

	mLeaf		= leaf;
	leaf->mNode = weak_from_this();
	if (graph) graph->registerLeaf(leaf);

	mUpdateFlags |= UpdateFlags::Leaf;
	// A leaf update leads to a subgraph structure change.
	propagateUpdateFlags(UpdateFlags::SubgraphStructure);
	propagateContentFlags(leaf->getContentFlags());
}

void SceneGraphNode::updateLocalTransform() {
	mLocalTransform = Affine3f().translate(mTranslation).rotate(mRotation).scale(mScaling);
}

void SceneGraphNode::propagateUpdateFlags(UpdateFlags flags) { 
	SceneGraphWalker walker(this, nullptr);
	while (walker) {
		walker->mUpdateFlags |= flags;	
		walker.up();
	}
}

void SceneGraphNode::propagateContentFlags(ContentFlags flags) {
	SceneGraphWalker walker(this, nullptr);
	while (walker) {
		walker->mContentFlags = flags;
		/*aggregate children contents*/
		SceneGraphWalker child(walker->getFirstChild(), walker.get());
		while (child) {
			walker->mContentFlags |= child->mContentFlags;
			child.next(false);
		}
		walker.up();
	}
}

void SceneGraphNode::reverseChildren() {
	// in-place linked list inversion...
	std::shared_ptr<SceneGraphNode> current, prev, next;
	current = mFirstChild;
	while (current) {
		next				  = current->mNextSibling;
		current->mNextSibling = prev;
		prev				  = current;
		current				  = next;
	}
	mFirstChild = prev;
}

int SceneGraphWalker::next(bool allowChildren) {
	// if allowChildren == true, then next() follows DFS traversal order.
	if (!mCurrent) return 0;
	if (allowChildren) {
		auto firstChild = mCurrent->getFirstChild();
		if (firstChild) {
			mCurrent = firstChild;
			return 1;
		}
	}
	int depth = 0;
	while (mCurrent) {
		if (mCurrent == mScope) {
			mCurrent = nullptr;
			return depth;
		}
		auto nextSibling = mCurrent->getNextSibling();
		if (nextSibling) {
			mCurrent = nextSibling;
			return depth;
		}
		mCurrent = mCurrent->getParent();
		--depth;
	}
	return depth;
}

int SceneGraphWalker::up() {
	if (mCurrent == mScope) mCurrent = nullptr;
	if (!mCurrent) return 0;
	mCurrent = mCurrent->getParent();
	return -1;
}

bool SceneAnimationChannel::apply(float time) const { 
	auto node = mTargetNode.lock();
	if (!node) {
		Log(Warning, "Animation channel has no target node");
		return false;
	}

	auto valueOptional = mSampler->evaluate(time);
	if (!valueOptional.has_value()) return false;
	Array4f value = valueOptional.value();

	switch (mAttribute) { 
	case anime::AnimationAttribute::Scaling:
		node->setScaling(Vector3f(value.head<3>()));
		break;
	case anime::AnimationAttribute::Rotation: {
		Quaternionf quat = Quaternionf(value.w(), value.x(), value.y(), value.z());
		if( quat.norm() != 0 ) node->setRotation(quat.normalized());
		else Log(Warning, "Rotation quaternion is zero, skipping rotation update");
		break;
	}
	case anime::AnimationAttribute::Translation:
		node->setTranslation(Vector3f(value.head<3>()));
		break;
	case anime::AnimationAttribute::Undefined:
	default:
		Log(Warning, "Undefined animation attribute");
		return false;
	}
	return true; 
}

bool SceneAnimation::apply(float time) const {
	bool success = true;
	for (const auto &channel : mChannels) 
		success &= channel->apply(time);
	return success;
}

void SceneAnimation::addChannel(const SceneAnimationChannel::SharedPtr& channel) {
	mChannels.push_back(channel);
	mDuration = max(mDuration, channel->getSampler()->getEndTime());
}

SceneGraphNode::SharedPtr SceneGraph::setRoot(const SceneGraphNode::SharedPtr &root) {
	auto oldRoot = mRoot;
	if (oldRoot) detach(oldRoot);	// and unregister the resources...
	attach(nullptr, root);			// and register the resources of the new graph...
	return oldRoot;
}

SceneGraphNode::SharedPtr SceneGraph::attach(const SceneGraphNode::SharedPtr &parent,
											 const SceneGraphNode::SharedPtr &child) {
	SceneGraph::SharedPtr parentGraph = parent ? parent->mGraph.lock() : shared_from_this();
	SceneGraph::SharedPtr childGraph  = child->mGraph.lock();
	
	if (!parentGraph && !childGraph) {
		// neither nodes has belonged to a graph...
		DCHECK(parent);
		child->mNextSibling = parent->mFirstChild;
		child->mParent		= parent.get();
		parent->mFirstChild = child;
		return child;
	}

	DCHECK(parentGraph.get() == this);
	SceneGraphNode::SharedPtr attachedChild;

	if (childGraph) {
		// subgraph belongs to another graph, copy the entire subgraph
		SceneGraphNode *currentParent = parent.get();
		SceneGraphWalker walker(child.get());
		while (walker) {
			// copy each node and their topology
			SceneGraphNode::SharedPtr copy = std::make_shared<SceneGraphNode>();
			// the root of the copied subgraph...
			if (!attachedChild) attachedChild = copy;

			copy->mName					   = walker->mName;
			copy->mParent				   = currentParent;
			copy->mContentFlags			   = walker->mContentFlags;
			copy->mGraph				   = weak_from_this();
			if (walker->mHasLocalTransform) 
				copy->setTransform(&walker->getTranslation(), &walker->getRotation(),
								   &walker->getScaling());
			if (walker->mLeaf) 
				copy->setLeaf(walker->mLeaf->clone());
			if (currentParent) {
				copy->mNextSibling		   = parent->mFirstChild;
				currentParent->mFirstChild = copy;
			} else { // parent do not exist at the beginning...
				mRoot = copy;
			}

			if (!attachedChild) attachedChild = copy;

			int deltaDepth = walker.next(true);
			if (deltaDepth > 0) {
				currentParent = copy.get();
			} else {
				while (deltaDepth++ < 0) {
					// make the children linked list same as original...
					currentParent->reverseChildren();
					currentParent = currentParent->mParent;
				}
			}
		}

	} else {
		// attach an orphaned subgraph...
		SceneGraphWalker walker(child.get());
		while (walker) {
			walker->mGraph = weak_from_this();
			auto leaf	   = walker->getLeaf();
			if (leaf) registerLeaf(leaf);
			walker.next(true);
		}
		child->mParent = parent.get();
		if (parent) {
			child->mNextSibling = parent->mFirstChild;
			parent->mFirstChild = child;
		} else {
			// parent == nullptr, i.e. set root.
			mRoot = child;
		}
		attachedChild = child;
	}
	attachedChild->propagateContentFlags(attachedChild->getContentFlags());
	return attachedChild;
}

SceneGraphNode::SharedPtr SceneGraph::attachLeaf(const SceneGraphNode::SharedPtr &parent,
												 const SceneGraphLeaf::SharedPtr &leaf) {
	auto node = std::make_shared<SceneGraphNode>();
	if (leaf->getNode()) node->setLeaf(leaf->clone());
	else node->setLeaf(leaf);
	return attach(parent, node);
}

SceneGraphNode::SharedPtr SceneGraph::detach(const SceneGraphNode::SharedPtr& node) {
	SceneGraph::SharedPtr graph = node->mGraph.lock();
	if (graph) {
		DCHECK(graph.get() == this);
		SceneGraphWalker walker(node.get());
		while (walker) {
			// traverse the subgraph, unregister all leaves.
			walker->mGraph.reset();
			auto leaf = walker->getLeaf();
			if (leaf) unregisterLeaf(leaf);
			walker.next(true);
		}
		if (node->mParent) {
			node->mParent->propagateUpdateFlags(SceneGraphNode::UpdateFlags::SubgraphStructure);
			node->mParent->propagateContentFlags(SceneGraphLeaf::ContentFlags::None);
			SceneGraphNode::SharedPtr* sibling = &node->mParent->mFirstChild;
			// delete this node from the linked list.
			while (*sibling && *sibling != node)
				sibling = &(*sibling)->mNextSibling;
			if (*sibling) *sibling = node->mNextSibling;
		}

		node->mParent = nullptr;
		node->mNextSibling.reset();

		if (mRoot == node) {
			mRoot.reset();
			mRoot = std::make_shared<SceneGraphNode>();
		}

		node->mParent = nullptr;
	}
	return node;
}

void SceneGraph::addMaterial(Material::SharedPtr material) {
	material->mMaterialId = mMaterials.size();
	mMaterials.push_back(material);
	mRoot->mUpdateFlags |= SceneGraphNode::UpdateFlags::SubgraphContent;
}

void SceneGraph::addMesh(Mesh::SharedPtr mesh) {
	mesh->meshId = mMeshes.size();
	mMeshes.push_back(mesh);
	mRoot->mUpdateFlags |= SceneGraphNode::UpdateFlags::SubgraphContent;
}

void SceneGraph::registerLeaf(const SceneGraphLeaf::SharedPtr& leaf) {
	if (!leaf) {
		Log(Warning, "Attempting to register an empty leaf...");
		return;
	}

	if (auto meshInstance = std::dynamic_pointer_cast<MeshInstance>(leaf)) {
		const auto &mesh			 = meshInstance->getMesh();
		auto it						 = std::find(mMeshes.begin(), mMeshes.end(), mesh);
		if (it == mMeshes.end())
			Log(Error, "The leaf node points to a mesh that does not added to the scene");
		meshInstance->mInstanceId = mMeshInstances.size();
		mMeshInstances.push_back(meshInstance);
		mRoot->mUpdateFlags |= SceneGraphNode::UpdateFlags::SubgraphContent;
	} else if (auto animation = std::dynamic_pointer_cast<SceneAnimation>(leaf)) {
		mAnimations.push_back(animation);
	} else if (auto light = std::dynamic_pointer_cast<SceneLight>(leaf)) {
		mLights.push_back(light);
	} else if (auto medium = std::dynamic_pointer_cast<Volume>(leaf)) {
		mMedia.push_back(medium);
		mRoot->mUpdateFlags |= SceneGraphNode::UpdateFlags::SubgraphContent;
	}
}

void SceneGraph::unregisterLeaf(const SceneGraphLeaf::SharedPtr& leaf) {
	if (!leaf) {
		Log(Warning, "Attempting to unregister an empty leaf...");
		return;
	}
	// This assumes that a mesh instance should only be attached to one leaf.
	if (auto meshInstance = std::dynamic_pointer_cast<MeshInstance>(leaf)) {
		auto it = std::find(mMeshInstances.begin(), mMeshInstances.end(), leaf);
		if (it != mMeshInstances.end()) mMeshInstances.erase(it);
		else Log(Warning, "Unregistering a mesh instance that do not exist in graph...");
	} else if (auto animation = std::dynamic_pointer_cast<SceneAnimation>(leaf)) {
		auto it = std::find(mAnimations.begin(), mAnimations.end(), leaf);
		if (it != mAnimations.end()) mAnimations.erase(it);
		else Log(Warning, "Unregistering an animation that do not exist in graph...");
	} else if (auto light = std::dynamic_pointer_cast<SceneLight>(leaf)) {
		auto it = std::find(mLights.begin(), mLights.end(), leaf);
		if (it != mLights.end()) mLights.erase(it);
		else Log(Warning, "Unregistering a light that do not exist in graph...");
	} else if (auto medium = std::dynamic_pointer_cast<Volume>(leaf)) {
		auto it = std::find(mMedia.begin(), mMedia.end(), leaf);
		if (it != mMedia.end()) mMedia.erase(it);
		else Log(Warning, "Unregistering a medium that do not exist in graph...");
	}
}

void SceneGraph::update(size_t frameIndex) {
	PROFILE("Update scene graph");
	struct AncestorContext {
		// Used to determine whether the subgraph transform should be updated,
		// due to the updated transform of an ancestor node.
		bool superGraphTransformUpdated = false;
	};

	if (mRoot->getUpdateFlags() != SceneGraphNode::UpdateFlags::None)
		mLastUpdateRecord = {frameIndex, mRoot->getUpdateFlags()};
	bool hasPendingStructureChanges =
		mRoot && (mRoot->getUpdateFlags() & (SceneGraphNode::UpdateFlags::SubgraphStructure |
											 SceneGraphNode::UpdateFlags::SubgraphContent)) != 0;

	std::vector<AncestorContext> stack;	// stack of ancestor nodes.
	AncestorContext context{};			// current context.
	SceneGraphWalker walker(mRoot.get());

	while (walker) {
		auto current = walker.get();
		auto parent = current->getParent();

		// update the transform of the node.
		bool currentTransformUpdated =
			(current->getUpdateFlags() & SceneGraphNode::UpdateFlags::LocalTransform) != 0;

		if (currentTransformUpdated) {
			// update the local transform matrix using SRT.
			current->updateLocalTransform();
		}

		if (parent) {
			// update the global transform using parent's global transform.
			current->mGlobalTransform = current->mHasLocalTransform
				? parent->getGlobalTransform() * current->getLocalTransform()
				: parent->getGlobalTransform();
		} else {
			current->mGlobalTransform = current->getLocalTransform();
		}

		// initialize the global bbox of the current node, start with the leaf (or an empty box if
		// there is no leaf)
		if ((current->getUpdateFlags() & (SceneGraphNode::UpdateFlags::SubgraphStructure |
			SceneGraphNode::UpdateFlags::SubgraphTransform)) != 0 || context.superGraphTransformUpdated){
			current->mGlobalBoundingBox = AABB{};
			if (current->getLeaf()) {
				AABB localBoundingBox = current->getLeaf()->getLocalBoundingBox();	
				current->mGlobalBoundingBox = localBoundingBox.transformed(current->getGlobalTransform());
			}
		}

		// whether we need to go deeper into the subgraph?
		bool subgraphNeedsUpdate = (current->getUpdateFlags() | SceneGraphNode::UpdateFlags::SubgraphUpdates) != 0;
		subgraphNeedsUpdate |= context.superGraphTransformUpdated;
		// negative delta means going upper in the hierarchy.
		int deltaDepth = walker.next(subgraphNeedsUpdate);

		// refresh the update flags [TODO: add prevTransform state for motion data...]
		current->mUpdateFlags = SceneGraphNode::UpdateFlags::None;

		if (deltaDepth > 0) { // goes to a child node
			stack.push_back(context);
			context.superGraphTransformUpdated |= currentTransformUpdated;
		} else {	
			// either goes to a sibling, or a child.
			if (parent) {	// in either case, the parent's bbox needs to be updated.
				parent->mGlobalBoundingBox.extend(current->getGlobalBoundingBox());
				parent->mUpdateFlags = current->getUpdateFlags() & SceneGraphNode::UpdateFlags::SubgraphUpdates;
			}

			while (deltaDepth++ < 0) {
				DCHECK(parent);
				current = parent;
				parent	= current->getParent();

				if (parent) {
					// merge bounding box changes and propagate update flags
					parent->mGlobalBoundingBox.extend(current->getGlobalBoundingBox());
					parent->mUpdateFlags |=
						current->getUpdateFlags() & SceneGraphNode::UpdateFlags::SubgraphUpdates;
				}
				
				context = stack.back();
				stack.pop_back();
			}
		}
	}

	// Any changes to leave contents leads to a full update to the following...
	if (hasPendingStructureChanges) {
		// reindex the instances, meshes and materials (e.g. when new things are added)
		int instanceIndex = 0;
		for (MeshInstance::SharedPtr instance : mMeshInstances) {
			instance->mInstanceId = instanceIndex++;
		}
		int meshIndex = 0;
		for (Mesh::SharedPtr mesh : mMeshes) {
			mesh->meshId = meshIndex++;
		}
		int materialIndex = 0;
		for (Material::SharedPtr material : mMaterials) {
			material->mMaterialId = materialIndex++;
		}
		int mediumIndex = 0;
		for (Volume::SharedPtr medium : mMedia) {
			medium->mediumId = mediumIndex++;
		}
	}
}

void SceneGraph::animate(double currentTime) {
	for (const auto &animation : mAnimations) {
		float duration		= animation->getDuration();
		float animationTime = std::fmod(currentTime, duration);
		animation->apply(animationTime);
	}
}

void SceneLight::setPosition(const Vector3f& position) {
	if (auto node = getNode()) {
		auto globalTransform = Affine3f::Identity();
		if (auto parent = node->getParent())
			globalTransform = parent->getGlobalTransform();
		node->setTranslation(globalTransform.inverse() * position);
		setUpdated(true);
	} else {
		Log(Error, "Setting direction for a light that does not exist in scene graph.");
	}
}

Vector3f SceneLight::getPosition() const { 
	if (auto node = getNode())
		return node->getGlobalTransform().translation();
	else return Vector3f::Zero();
}

void SceneLight::setDirection(const Vector3f& direction) {
	if (auto node = getNode()) {
		Affine3f globalTransform = Affine3f::Identity();
		if (auto parent = node->getParent()) globalTransform = parent->getGlobalTransform();
		
		Affine3f worldToLocal  = look_at(Vector3f{}, direction, Vector3f{0, 1, 0});
		Affine3f localToParent = (worldToLocal * globalTransform).inverse();
		node->setRotation(Quaternionf(localToParent.rotation()));
		node->setScaling(localToParent.scaling());
		setUpdated(true);
	} else {
		Log(Error, "Setting direction for a light that does not exist in scene graph.");
	}
}

Vector3f SceneLight::getDirection() const {
	if (auto node = getNode())
		return node->getGlobalTransform().rotation() * -Vector3f::UnitZ();
	else return Vector3f::Zero();
}

void SceneLight::renderUI() { 
	auto direction = getDirection(), position = getPosition();
	auto color	= getColor();
	float scale = getScale();
	if (ui::DragFloat3("RGB", (float *) &color, 1e-3, 0, 1)) setColor(color);
	if (ui::DragFloat("Scale", &scale, 1e-2, 0, 100)) setScale(scale);
	if (ui::DragFloat3("Direction input", (float *) &direction, 1e-3, -1, 1)) setDirection(direction);
	if (ui::DragFloat3("Position", (float *) &position), 1e-2, -100, 100) setPosition(position);
}

void DirectionalLight::renderUI() {
	RGB color	= getColor();
	float scale = getScale();
	Vector2f sphericalDir = worldToLatLong(getDirection());
	if (ui::DragFloat3("RGB", (float *) &color, 1e-3, 0, 1)) setColor(color);
	if (ui::DragFloat("Scale", &scale, 1e-2, 0, 100)) setScale(scale);
	if(ui::SliderFloat("Azimuth", &sphericalDir[0], 0, 1) ||
		ui::SliderFloat("Altitude", &sphericalDir[1], 0, 1))
		setDirection(latlongToWorld(sphericalDir));
}

void SceneAnimationChannel::renderUI() {
	if (!isValid()) return;
	ui::Text("Duration: %f - %f", getSampler()->getStartTime(), getSampler()->getEndTime());
	ui::Text("Key frames: %zd", getSampler()->getKeyframes().size());
	ui::Text("Target node: %s", getTargetNode()->getName().c_str());
}

void MeshInstance::renderUI() {
	if (mMesh) {
		ui::BulletText("Target mesh");
		if (ui::TreeNode(mMesh->getName().c_str())){
			mMesh->renderUI();
			ui::TreePop();
		}
	}
}

void SceneAnimation::renderUI() {
	ui::Text("End time: %f", getDuration());
	ui::Text("Channels: ");
	static const auto getChannelName = [](const SceneAnimationChannel::SharedPtr &channel)
	-> std::string{
		if (!channel->isValid()) return "<Invalid Channel>";
		std::string name = channel->getTargetNode()->getName();
		auto attribute	 = channel->getAttribute();
		if (attribute == anime::AnimationAttribute::Scaling) name += " <S>";
		else if (attribute == anime::AnimationAttribute::Rotation) name += " <R>";
		else if (attribute == anime::AnimationAttribute::Translation) name += " <T>";
		return name;
	};
	for (auto channel : mChannels) {
		if (channel->isValid()) {
			channel->getAttribute();
			if (ui::TreeNode(getChannelName(channel).c_str())) {
				channel->renderUI();
				ui::TreePop();
			}
		}
	}
}

void HomogeneousVolume::renderUI() { 
	ui::Text("Homogeneous Volume"); 
	ui::Text(("Sigma_a: " + sigma_t.string()).c_str());
	ui::Text(("Sigma_s: " + albedo.string()).c_str());
	ui::Text("g: %f", g);
	if (isEmissive()) ui::Text("Le: %s", Le.string().c_str());
}

void VDBVolume::renderUI() {
	ui::Text("OpenVDB Volume Data");
	ui::Text(("Sigma_a: " + sigma_t.string()).c_str());
	ui::Text(("Sigma_s: " + albedo.string()).c_str());
	ui::Text("g: %f", g);
}

void SceneGraphNode::renderUI() { 
	if (mHasLocalTransform && ui::TreeNode("Transformation")) {
		if (getScaling() != Vector3f::Ones())
			ui::Text(("Scaling:\n" + getScaling().string()).c_str());
		if (getRotation() != Quaternionf::Identity())
			ui::Text(("Rotation:\n" + getRotation().string()).c_str());
		if (getTranslation() != Vector3f::Zero())
			ui::Text(("Translation:\n" + getTranslation().string()).c_str());
		ui::TreePop();
	}
	if (ui::TreeNode("Bounding box")) {
		ui::Text(("Min: " + getGlobalBoundingBox().min().string()).c_str());
		ui::Text(("Max: " + getGlobalBoundingBox().max().string()).c_str());
		ui::TreePop();
	}
	std::string contents;
	if (int(getContentFlags() & SceneGraphNode::ContentFlags::Mesh)) contents += "Mesh Instance, ";
	if (int(getContentFlags() & SceneGraphNode::ContentFlags::Animation)) contents += "Animation, ";
	if (int(getContentFlags() & SceneGraphNode::ContentFlags::Light)) contents += "Light, ";
	if (int(getContentFlags() & SceneGraphNode::ContentFlags::Volume)) contents += "Volume, ";
	ui::Text(("Contents: " + contents).c_str());
	if (getLeaf()) {
		ui::Text("Leaf:");
		if (auto instance = std::dynamic_pointer_cast<MeshInstance>(getLeaf())) {
			if (ui::TreeNode("Mesh Instance")) {
				ui::PushID(instance->getInstanceId());
				instance->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		} else if (auto animation = std::dynamic_pointer_cast<SceneAnimation>(getLeaf())) {
			if (ui::TreeNode("Animation")) {
				animation->renderUI();
				ui::TreePop();
			}
		} else if (auto light = std::dynamic_pointer_cast<SceneLight>(getLeaf())) {
			if (ui::TreeNode("Light")) {
				light->renderUI();
				ui::TreePop();
			}
		} else if (auto volume = std::dynamic_pointer_cast<Volume>(getLeaf())) {
			if (ui::TreeNode("Volume")) {
				volume->renderUI();
				ui::TreePop();
			}
		} else
			ui::Text("Unknown leaf type");
	}
	SceneGraphNode *child = getFirstChild();
	if (child) ui::Text("Children nodes");
	size_t childId = 0;
	while (child) {
		ui::PushID(childId++);
		if (ui::TreeNode(child->getName().empty() ?
			"Unnamed Node" : child->getName().c_str())) {
			child->renderUI();
			ui::TreePop();
		}
		ui::PopID();
		child = child->getNextSibling();
	}
}

void SceneGraph::renderUI() {
	if (mRoot && ui::TreeNode(mRoot->getName().empty() ? 
		"Root" : mRoot->getName().c_str())) {
		mRoot->renderUI();
		ui::TreePop();
	}
}

KRR_NAMESPACE_END