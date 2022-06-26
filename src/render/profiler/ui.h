#pragma once
#include "profiler.h"

struct ImVec2;

KRR_NAMESPACE_BEGIN

class ProfilerUI {
public:
	using UniquePtr = std::unique_ptr<ProfilerUI>;

	enum GraphMode : uint32_t{
		Off,
		CpuTime,
		GpuTime,
		Count,
	};

	/** Create a profiler UI instance.
		\param[in] pProfiler Profiler.
	*/
	static UniquePtr create(const Profiler::SharedPtr& pProfiler);

	/** Render the profiler UI.
		Note: This must be called within a valid ImGui window.
	*/
	void render();

private:
	ProfilerUI(const Profiler::SharedPtr& pProfiler) : mpProfiler(pProfiler) {}

	/** Render the profiler options.
	*/
	void renderOptions();

	/** Render the graph.
		\param[in] size Size of the graph in pixels.
		\param[in] highlightIndex Highlighted event index.
		\param[out] newHighlightIndex New highlighted event index (unchanged if mouse not over graph).
	*/
	void renderGraph(const ImVec2& size, size_t highlightIndex, size_t& newHighlightIndex);

	/** Update the internal event data from the current profiler event data.
	*/
	void updateEventData();

	/** Update the graph data.
		Retrieves the new current value for the graph depending on the current graph mode
		and writes to the graph history.
	*/
	void updateGraphData();

	/** Clear the graph data.
	*/
	void clearGraphData();

	Profiler::SharedPtr mpProfiler;         ///< Profiler instance.

	GraphMode mGraphMode = GraphMode::Off;  ///< Graph mode.
	bool mEnableAverage = true;             ///< Use averaged time values (EMA).

	struct EventData {
		Profiler::Event* pEvent;            ///< Source profiler event.
		std::string name;                   ///< Short name.
		uint32_t level;                     ///< Event tree level.
		uint32_t color;                     ///< Color (for graph).
		uint32_t mutedColor;                ///< Muted color (for graph).
		float cpuTime;                      ///< Current CPU time.
		float gpuTime;                      ///< Current GPU time.

		// Graph data.
		float graphValue;                   ///< Current graph value.
		float maxGraphValue;                ///< Maximum graph value in history.
		std::vector<float> graphHistory;    ///< Graph value history.
	};

	std::vector<EventData> mEventData;      ///< Preprocessed event data (presisted across frames).

	float mTotalCpuTime = 0.f;              ///< Total CPU time of top level events.
	float mTotalGpuTime = 0.f;              ///< Total GPU time of top level events.
	float mTotalGraphValue = 0.f;           ///< Total graph value of top level events.

	size_t mHistoryLength = 0;              ///< Current length of graph history.
	size_t mHistoryWrite = 0;               ///< Graph history write index (round-robin).

	size_t mHighlightIndex = -1;            ///< Highlighted event index.
};

KRR_NAMESPACE_END