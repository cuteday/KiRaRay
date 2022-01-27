#include <optix_function_table_definition.h>
#include <optix_types.h>

#include "renderer.h"

KRR_NAMESPACE_BEGIN

extern "C" char PTX_CODE[];

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	int objectID;

	vec3f* vertices;
	vec3i* indices;
	vec3f color;
};

Renderer::Renderer() {
	initOptix();

	logInfo("#krr: creating optix context ...");
	createContext();

	logInfo("#krr: setting up module ...");
	createModule();

	logInfo("#krr: creating raygen programs ...");
	createRaygenPrograms();
	logInfo("#krr: creating miss programs ...");
	createMissPrograms();

	logInfo("#krr: creating hitgroup programs ...");
	createHitgroupPrograms();

	logInfo("#krr: setting up optix pipeline ...");
	createPipeline();
	
	launchParamsBuffer.alloc(sizeof(launchParams));

	logInfo("#krr: context, module, pipeline, etc, all set up ...");
	logSuccess("Renderer::Optix 7 context fully set up");
}

void Renderer::initOptix()
{
	std::cout << "#krr: initializing optix..." << std::endl;

	// -------------------------------------------------------
	// check for available optix7 capable devices
	// -------------------------------------------------------
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw std::runtime_error("#krr: no CUDA capable devices found!");
	std::cout << "#krr: found " << numDevices << " CUDA devices" << std::endl;

	// -------------------------------------------------------
	// initialize optix
	// -------------------------------------------------------
	OPTIX_CHECK(optixInit());
}

static void context_log_cb(unsigned int level,
	const char* tag,
	const char* message,
	void*)
{
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

/*! creates and configures a optix device context (in this simple
	example, only for the primary GPU device) */
void Renderer::createContext()
{
	// for this sample, do everything on one device
	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));
	CUDA_CHECK(cudaStreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	std::cout << "#krr: running on device: " << deviceProps.name << std::endl;

	CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &mOptixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback
	(mOptixContext, context_log_cb, nullptr, 4));
}



/*! creates the module that contains all the programs we are going
	to use. in this simple example, we use a single module from a
	single .cu file, using a single embedded ptx string */
void Renderer::createModule()
{
	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions = {};
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	pipelineLinkOptions.maxTraceDepth = 2;

	const std::string ptxCode = PTX_CODE;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixModuleCreateFromPTX(mOptixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptxCode.c_str(),
		ptxCode.size(),
		log, &sizeof_log,
		&module
	));
	if (sizeof_log > 1) PRINT(log);
}



/*! does all setup for the raygen program(s) we are going to use */
void Renderer::createRaygenPrograms()
{
	// we do a single ray gen program in this example:
	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(mOptixContext,
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&raygenPGs[0]
	));
	if (sizeof_log > 1) PRINT(log);
}

/*! does all setup for the miss program(s) we are going to use */
void Renderer::createMissPrograms()
{
	// we do a single ray gen program in this example:
	missPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;
	pgDesc.miss.entryFunctionName = "__miss__radiance";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(mOptixContext,
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&missPGs[0]
	));
	if (sizeof_log > 1) PRINT(log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void Renderer::createHitgroupPrograms()
{
	// for this simple example, we set up a single hit group
	hitgroupPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(mOptixContext,
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&hitgroupPGs[0]
	));

	if (sizeof_log > 1) PRINT(log);
}


/*! assembles the full pipeline of all programs */
void Renderer::createPipeline()
{
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(mOptixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		programGroups.data(),
		(int)programGroups.size(),
		log, &sizeof_log,
		&pipeline
	));
	if (sizeof_log > 1) PRINT(log);

	OPTIX_CHECK(optixPipelineSetStackSize
	(/* [in] The pipeline to configure the stack size for */
		pipeline,
		/* [in] The direct stack size requirement for direct
		   callables invoked from IS or AH. */
		2 * 1024,
		/* [in] The direct stack size requirement for direct
		   callables invoked from RG, MS, or CH.  */
		2 * 1024,
		/* [in] The continuation stack requirement. */
		2 * 1024,
		/* [in] The maximum depth of a traversable graph
		   passed to trace. */
		1));
	if (sizeof_log > 1) PRINT(log);
}


/*! constructs the shader binding table */
void Renderer::buildSBT()
{
	// ------------------------------------------------------------------
	// build raygen records
	// ------------------------------------------------------------------
	std::vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
	}
	raygenRecordsBuffer.alloc_and_copy_from_host(raygenRecords);
	sbt.raygenRecord = raygenRecordsBuffer.data();

	// ------------------------------------------------------------------
	// build miss records
	// ------------------------------------------------------------------
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}
	missRecordsBuffer.alloc_and_copy_from_host(missRecords);
	sbt.missRecordBase = missRecordsBuffer.data();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();

	// ------------------------------------------------------------------
	// build hitgroup records
	// ------------------------------------------------------------------

	// we don't actually have any objects in this example, but let's
	// create a dummy one so the SBT doesn't have any null pointers
	// (which the sanity checks in compilation would complain about)
	uint numMeshes = mpScene->meshes.size();
	std::vector<HitgroupRecord> hitgroupRecords;
	for (uint i = 0; i < numMeshes; i++) {
		int objectType = 0;
		HitgroupRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
		rec.objectID = i;
		rec.vertices = (vec3f*)vertexBuffers[i].data();
		rec.indices = (vec3i*)indexBuffers[i].data();
		hitgroupRecords.push_back(rec);
	}
	hitgroupRecordsBuffer.alloc_and_copy_from_host(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.data();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void Renderer::buildAS()
{
	std::vector<Mesh>& meshes = mpScene->meshes;

	assert(indexBuffers.size() == 0);
	assert(vertexBuffers.size() == 0);
	indexBuffers.resize(meshes.size());
	vertexBuffers.resize(meshes.size());

	std::vector<OptixBuildInput> triangleInputs(meshes.size());
	std::vector<uint> triangleInputFlags(meshes.size());
	std::vector<CUdeviceptr> vertexBufferPtr(meshes.size());
	std::vector<CUdeviceptr> indexBufferPtr(meshes.size());

	for (uint i = 0; i < meshes.size(); i++) {
		Mesh& mesh = meshes[i];

		indexBuffers[i].alloc_and_copy_from_host(mesh.indices);
		vertexBuffers[i].alloc_and_copy_from_host(mesh.vertices);

		indexBufferPtr[i] = indexBuffers[i].data();
		vertexBufferPtr[i] = vertexBuffers[i].data();

		triangleInputs[i] = {};
		triangleInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		// vertex data desc
		triangleInputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInputs[i].triangleArray.vertexStrideInBytes = sizeof(vec3f);
		triangleInputs[i].triangleArray.numVertices = mesh.vertices.size();
		triangleInputs[i].triangleArray.vertexBuffers = &vertexBufferPtr[i];
		// index data desc
		triangleInputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInputs[i].triangleArray.indexStrideInBytes = sizeof(vec3i);
		triangleInputs[i].triangleArray.numIndexTriplets = mesh.indices.size();
		triangleInputs[i].triangleArray.indexBuffer = indexBufferPtr[i];

		triangleInputFlags[i] = 0;

		triangleInputs[i].triangleArray.flags = &triangleInputFlags[i];
		triangleInputs[i].triangleArray.numSbtRecords = 1;
		triangleInputs[i].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInputs[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInputs[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE |
		OPTIX_BUILD_FLAG_ALLOW_COMPACTION
		;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(mOptixContext,
		&accelOptions, triangleInputs.data(), meshes.size(), &blasBufferSizes));

	// prepare for compaction
	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.data();

	// building process
	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

	OptixTraversableHandle &asHandle = launchParams.traversable;
	
	OPTIX_CHECK(optixAccelBuild(mOptixContext,
		stream,
		&accelOptions,
		triangleInputs.data(),
		meshes.size(),
		tempBuffer.data(),
		tempBuffer.size(),
		outputBuffer.data(),
		outputBuffer.size(),
		&asHandle,
		&emitDesc,
		1));
	CUDA_SYNC_CHECK();

	// perform compaction
	uint64_t compactedSize;
	compactedSizeBuffer.copy_to_host(&compactedSize, 1);
	
	accelBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(mOptixContext,
		stream,
		asHandle,
		accelBuffer.data(),
		accelBuffer.size(),
		&asHandle));
	CUDA_SYNC_CHECK();

	// clean up...
}

void Renderer::onKeyEvent(const KeyboardEvent& keyEvent)
{
	if (mpScene) mpScene->onKeyEvent(keyEvent);
}

void Renderer::onMouseEvent(const MouseEvent& mouseEvent)
{
	if (mpScene) mpScene->onMouseEvent(mouseEvent);
}

void Renderer::renderUI() {
	mpScene->renderUI();
}

/*! render one frame */
void Renderer::render()
{
	if (launchParams.fbSize.x * launchParams.fbSize.y == 0) return;

	CUDA_SYNC_CHECK();

	launchParamsBuffer.copy_from_host(&launchParams, 1);
	launchParams.frameID++;

	OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
		pipeline, stream,
		/*! parameters and SBT */
		launchParamsBuffer.data(),
		launchParamsBuffer.size(),
		&sbt,
		/*! dimensions of the launch: */
		launchParams.fbSize.x,
		launchParams.fbSize.y,
		1
	));
	// sync - make sure the frame is rendered before we download and
	// display (obviously, for a high-performance application you
	// want to use streams and double-buffering, but for this simple
	// example, this will have to do)

	CUDA_SYNC_CHECK();
}

/*! resize frame buffer to given resolution */
void Renderer::resize(const vec2i& size)
{
	// if window minimized
	if (size.x * size.y <= 0) return;

	// resize our cuda frame buffer
	colorBuffer.resize(size.x * size.y * sizeof(vec4f));

	// update the launch parameters that we'll pass to the optix
	// launch:
	launchParams.fbSize = size;
	launchParams.colorBuffer = (vec4f*)colorBuffer.data();
}

/*! download the rendered color buffer */
CUDABuffer& Renderer::result()
{
	return colorBuffer;
}

KRR_NAMESPACE_END