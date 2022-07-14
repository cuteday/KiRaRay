#include <optional>

#include "ui.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

namespace
{
	const char* kGraphModes[(size_t)ProfilerUI::GraphMode::Count] = { "Off", "CPU Time", "GPU Time" };

	const float kIndentWidth = 16.f;
	const float kPadding = 16.f;
	const float kBarWidth = 50.f;
	const float kBarHeight = 8.f;
	const uint32_t kBarColor = 0xffffffff;
	const uint32_t kBarMutedColor = 0xff404040;
	const float kGraphBarWidth = 2.f;

	static constexpr size_t kColumnCount = 5;
	const char* kColumnTitle[kColumnCount] = { "Event", "CPU Time", "CPU %%", "GPU Time", "GPU %%" };
	const float kHeaderSpacing = 5.f;

	const size_t kHistoryCapacity = 256;

	// Colorblind friendly palette.
	const std::vector<uint32_t> kColorPalette = {
		IM_COL32(0x00, 0x49, 0x49, 0xff),
		IM_COL32(0x00, 0x92, 0x92, 0xff),
		IM_COL32(0xff, 0x6d, 0xb6, 0xff),
		IM_COL32(0xff, 0xb6, 0xdb, 0xff),
		IM_COL32(0x49, 0x00, 0x92, 0xff),
		IM_COL32(0x00, 0x6d, 0xdb, 0xff),
		IM_COL32(0xb6, 0x6d, 0xff, 0xff),
		IM_COL32(0x6d, 0xb6, 0xff, 0xff),
		IM_COL32(0xb6, 0xdb, 0xff, 0xff),
		IM_COL32(0x92, 0x00, 0x00, 0xff),
		IM_COL32(0x24, 0xff, 0x24, 0xff),
		IM_COL32(0xff, 0xff, 0x6d, 0xff)
	};

	const uint32_t kHighlightColor = IM_COL32(0xff, 0x7f, 0x00, 0xcf);

	void drawRectFilled(const ImVec2& pos, const ImVec2& size, uint32_t color = 0xffffffff) {
		auto cursorPos = ui::GetCursorScreenPos();
		ui::GetWindowDrawList()->AddRectFilled(ImVec2(cursorPos[0] + pos[0], cursorPos[1] + pos[1]), ImVec2(cursorPos[0] + pos[0] + size[0], cursorPos[1] + pos[1] + size[1]), color);
	};

	void drawBar(float fraction, const ImVec2& size, ImU32 color = 0xffffffff, ImU32 background = 0x00000000, bool highlight = false) {
		auto cursorPos = ui::GetCursorScreenPos();
		auto height = ui::GetTextLineHeightWithSpacing();
		cursorPos[1] += 0.5f * (height - size[1]);
		ui::GetWindowDrawList()->AddRectFilled(ImVec2(cursorPos[0], cursorPos[1]), ImVec2(cursorPos[0] + size[0], cursorPos[1] + size[1]), background);
		ui::GetWindowDrawList()->AddRectFilled(ImVec2(cursorPos[0], cursorPos[1]), ImVec2(cursorPos[0] + fraction * size[0], cursorPos[1] + size[1]), color);
		if (highlight) ui::GetWindowDrawList()->AddRect(ImVec2(cursorPos[0], cursorPos[1]), ImVec2(cursorPos[0] + size[0], cursorPos[1] + size[1]), kHighlightColor);
	}
}

ProfilerUI::UniquePtr ProfilerUI::create(const Profiler::SharedPtr& pProfiler) {
	return UniquePtr(new ProfilerUI(pProfiler));
}

void ProfilerUI::render() {
	updateEventData();
	updateGraphData();

	renderOptions();

	// Compute column widths.
	float columnWidth[kColumnCount];
	for (size_t i = 0; i < kColumnCount; ++i) columnWidth[i] = ui::CalcTextSize(kColumnTitle[i])[0] + kPadding;
	for (const auto& eventData : mEventData) {
		columnWidth[0] = std::max(columnWidth[0], ui::CalcTextSize(eventData.name.c_str())[0] + eventData.level * kIndentWidth + kPadding);
	}

	float startY;
	float endY;
	size_t newHighlightIndex = -1;

	// Draw table (last column is used for graph).
	ui::Columns(kColumnCount + 1, "#events", false);
	for (size_t col = 0; col < kColumnCount; ++col) {
		ui::SetColumnWidth((int)col, columnWidth[col]);
		ui::Text(kColumnTitle[col]);
		ui::Dummy(ImVec2(0.f, kHeaderSpacing));

		if (col == 0) startY = ui::GetCursorPosY();

		for (size_t i = 0; i < mEventData.size(); ++i) {
			const auto& eventData = mEventData[i];
			auto pEvent = eventData.pEvent;

			if (col == 0) {// Event name
				float indent = eventData.level * kIndentWidth;
				if (indent > 0.f) ui::Indent(indent);
				auto color = (i == mHighlightIndex) ? ImColor(kHighlightColor) : ImColor(ui::GetStyleColorVec4(ImGuiCol_Text));
				ui::TextColored(color, eventData.name.c_str());
				if (indent > 0.f) ui::Unindent(indent);
			}
			else if (col == 1) {// CPU time
				ui::Text("%.2f ms", eventData.cpuTime);
				if (ui::IsItemHovered())
				{
					auto stats = eventData.pEvent->computeCpuTimeStats();
					ui::BeginTooltip();
					ui::Text("%s\nMin: %.2f\nMax: %.2f\nMean: %.2f\nStdDev: %.2f", eventData.name.c_str(), stats.min, stats.max, stats.mean, stats.stdDev);
					ui::EndTooltip();
				}
			}
			else if (col == 2) {// CPU %
				ui::PushID((int)reinterpret_cast<intptr_t>(&eventData));
				float fraction = mTotalCpuTime > 0.f ? eventData.cpuTime / mTotalCpuTime : 0.f;
				bool isGraphShown = mGraphMode == GraphMode::CpuTime;
				bool isHighlighted = isGraphShown && i == mHighlightIndex;
				drawBar(fraction, ImVec2(kBarWidth, kBarHeight), isGraphShown ? eventData.color : kBarColor, isGraphShown ? eventData.mutedColor : kBarMutedColor, isHighlighted);
				ui::Dummy(ImVec2(kBarWidth, ui::GetTextLineHeight()));
				if (ui::IsItemHovered()) {
					if (isGraphShown) newHighlightIndex = i;
					ui::BeginTooltip();
					ui::Text("%s\n%.1f%%", eventData.name.c_str(), fraction * 100.f);
					ui::EndTooltip();
				}
				ui::PopID();
			}
			else if (col == 3) { // GPU time
				ui::Text("%.2f ms", eventData.gpuTime);
				if (ui::IsItemHovered()) {
					auto stats = eventData.pEvent->computeGpuTimeStats();
					ui::BeginTooltip();
					ui::Text("%s\nMin: %.2f\nMax: %.2f\nMean: %.2f\nStdDev: %.2f", eventData.name.c_str(), stats.min, stats.max, stats.mean, stats.stdDev);
					ui::EndTooltip();
				}
			}
			else if (col == 4) { // GPU % (last column)
				ui::PushID((int)reinterpret_cast<intptr_t>(&eventData));
				float fraction = mTotalGpuTime > 0.f ? eventData.gpuTime / mTotalGpuTime : 0.f;
				bool isGraphShown = mGraphMode == GraphMode::GpuTime;
				bool isHighlighted = isGraphShown && i == mHighlightIndex;
				drawBar(fraction, ImVec2(kBarWidth, kBarHeight), isGraphShown ? eventData.color : kBarColor, isGraphShown ? eventData.mutedColor : kBarMutedColor, isHighlighted);
				ui::Dummy(ImVec2(kBarWidth, ui::GetTextLineHeight()));
				if (ui::IsItemHovered()) {
					if (isGraphShown) newHighlightIndex = i;
					ui::BeginTooltip();
					ui::Text("%s\n%.1f%%", eventData.name.c_str(), fraction * 100.f);
					ui::EndTooltip();
				}
				ui::PopID();
			}
		}

		if (col == 0) endY = ui::GetCursorPosY(); 
		ui::NextColumn();
	}

	// Set new highlight index if mouse is over one of the bars.
	if (newHighlightIndex != -1) mHighlightIndex = newHighlightIndex;

	// Draw the graph.
	if (mGraphMode != GraphMode::Off) {
		ui::Text("Graph");
		ui::Dummy(ImVec2(0.f, kHeaderSpacing));
		ImVec2 graphSize(ui::GetWindowSize()[0] - ui::GetCursorPosX(), endY - startY);
		renderGraph(graphSize, mHighlightIndex, newHighlightIndex);
		mHighlightIndex = newHighlightIndex;
	}
}

void ProfilerUI::renderOptions() {
	bool paused = mpProfiler->isPaused();
	if (ui::Checkbox("Pause", &paused)) mpProfiler->setPaused(paused);

	ui::SameLine();
	ui::Checkbox("Average", &mEnableAverage);

	ui::SameLine();
	ui::SetNextItemWidth(100.f);
	if (ui::Combo("Graph", reinterpret_cast<int*>(&mGraphMode), kGraphModes, (int)GraphMode::Count)) clearGraphData();

	if (mpProfiler->isCapturing()) {
		ui::SameLine();
		if (ui::Button("End Capture")) {
			auto pCapture = mpProfiler->endCapture();
			assert(pCapture);
		}
	}
	else {
		ui::SameLine();
		if (ui::Button("Start Capture")) mpProfiler->startCapture();
	}

	ui::Separator();
}

void ProfilerUI::renderGraph(const ImVec2& size, size_t highlightIndex, size_t& newHighlightIndex)
{
	ImVec2 mousePos = ui::GetMousePos();
	ImVec2 screenPos = ui::GetCursorScreenPos();
	mousePos[0] -= screenPos[0];
	mousePos[1] -= screenPos[1];

	float totalMaxGraphValue = 0.f;
	for (const auto& eventData : mEventData) totalMaxGraphValue += eventData.level == 0 ? eventData.maxGraphValue : 0.f;

	const float scaleY = size[1] / totalMaxGraphValue;

	float x = 0.f;
	float levelY[128];
	std::optional<float> highlightValue;

	for (size_t k = 0; k < mHistoryLength; ++k) {
		size_t historyIndex = (mHistoryWrite + kHistoryCapacity - k - 1) % kHistoryCapacity;

		float totalValue = 0.f;
		for (const auto& eventData : mEventData) {
			totalValue += eventData.level == 0 ? eventData.graphHistory[historyIndex] : 0.f;
		}

		levelY[0] = size[1] - totalValue * scaleY;
		float highlightY = 0.f;
		float highlightHeight = 0.f;

		for (size_t i = 0; i < mEventData.size(); ++i) {
			const auto& eventData = mEventData[i];
			float value = eventData.graphHistory[historyIndex];
			uint32_t level = eventData.level;

			float y = levelY[level];
			float height = value * scaleY;
			drawRectFilled(ImVec2(x, y), ImVec2(kGraphBarWidth, height), eventData.color);

			// Check if mouse is over this bar for tooltip value and highlighting in next frame.
			if (mousePos[0] >= x && mousePos[0] < x + kGraphBarWidth && mousePos[1] >= y && mousePos[1] < y + height) {
				newHighlightIndex = i;
				highlightValue = value / totalValue;
			}

			if (highlightIndex == i) {
				highlightY = y;
				highlightHeight = height;
			}

			levelY[level + 1] = levelY[level];
			levelY[level] += height;
		}

		if (highlightHeight > 0.f) drawRectFilled(ImVec2(x, highlightY), ImVec2(kGraphBarWidth, highlightHeight), kHighlightColor);

		x += kGraphBarWidth;
		if (x > size[0]) break;
	}

	ui::Dummy(ImVec2(size));
	if (ui::IsItemHovered() && highlightValue) {
		assert(newHighlightIndex >= 0 && newHighlightIndex < mEventData.size());
		ui::BeginTooltip();
		ui::Text("%s\n%.2f%%", mEventData[newHighlightIndex].name.c_str(), *highlightValue * 100.f);
		ui::EndTooltip();
	}
}

void ProfilerUI::updateEventData() {
	const auto& events = mpProfiler->getEvents();

	mEventData.resize(events.size());
	mTotalCpuTime = 0.f;
	mTotalGpuTime = 0.f;

	for (size_t i = 0; i < mEventData.size(); ++i) {
		auto& event = mEventData[i];
		auto pEvent = events[i];

		event.pEvent = pEvent;

		// Update name and level.
		std::string name = pEvent->getName();
		uint32_t level = std::max((uint32_t)std::count(name.begin(), name.end(), '/'), 1u) - 1;
		name = name.substr(name.find_last_of("/") + 1);
		event.name = name;
		event.level = level;

		// Use colors from color palette.
		event.color = kColorPalette[i % kColorPalette.size()];
		event.mutedColor = (event.color & 0xffffff) | 0x1f000000;

		// Get event times.
		event.cpuTime = mEnableAverage ? pEvent->getCpuTimeAverage() : pEvent->getCpuTime();
		event.gpuTime = mEnableAverage ? pEvent->getGpuTimeAverage() : pEvent->getGpuTime();

		// Sum up times.
		if (level == 0) {
			mTotalCpuTime += event.cpuTime;
			mTotalGpuTime += event.gpuTime;
		}
	}
}

void ProfilerUI::updateGraphData() {
	if (mGraphMode == GraphMode::Off) return;

	mTotalGraphValue = 0.f;

	for (auto& event : mEventData) {
		switch (mGraphMode) {
		case GraphMode::Off:
			continue;
		case GraphMode::CpuTime:
			event.graphValue = event.cpuTime;
			break;
		case GraphMode::GpuTime:
			event.graphValue = event.gpuTime;
			break;
		}

		if (event.level == 0) mTotalGraphValue += event.graphValue;

		event.graphHistory.resize(kHistoryCapacity);
		event.graphHistory[mHistoryWrite] = event.graphValue;

		float maxGraphValue = 0.f;
		for (size_t j = 0; j < mHistoryLength; ++j) {
			maxGraphValue = std::max(maxGraphValue, event.graphHistory[j]);
		}
		event.maxGraphValue = maxGraphValue;
	}

	if (!mpProfiler->isPaused()) {
		mHistoryWrite = (mHistoryWrite + 1) % kHistoryCapacity;
		mHistoryLength = std::min(mHistoryLength + 1, kHistoryCapacity);
	}
}

void ProfilerUI::clearGraphData() {
	mHistoryLength = 0;
	mHistoryWrite = 0;

	for (auto& event : mEventData) {
		event.graphValue = 0.f;
		event.maxGraphValue = 0.f;
	}
}

KRR_NAMESPACE_END