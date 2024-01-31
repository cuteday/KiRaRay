#pragma once

#define GLFW_INCLUDE_NONE // Do not include any OpenGL headers
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif // _WIN32
#include <GLFW/glfw3native.h>

#include "common.h"

NAMESPACE_BEGIN(krr)

namespace io {
	/** Input modifiers used with some events
	*/
	struct InputModifiers
	{
		bool isCtrlDown  = false;   ///< Any of the control keys are pressed
		bool isShiftDown = false;   ///< Any of the shift keys are pressed
		bool isAltDown   = false;   ///< Any of the alt keys are pressed
	};

	/** Abstracts mouse messages
	*/
	struct MouseEvent
	{
		/** Message Type
		*/
		enum class Type
		{
			LeftButtonDown,         ///< Left mouse button was pressed
			LeftButtonUp,           ///< Left mouse button was released
			MiddleButtonDown,       ///< Middle mouse button was pressed
			MiddleButtonUp,         ///< Middle mouse button was released
			RightButtonDown,        ///< Right mouse button was pressed
			RightButtonUp,          ///< Right mouse button was released
			Move,                   ///< Mouse cursor position changed
			Wheel                   ///< Mouse wheel was scrolled
		};

		Type type;              ///< Event Type.
		Vector2f pos;             ///< Normalized coordinates x,y in range [0, 1]. (0,0) is the top-left corner of the window.
		Vector2f screenPos;       ///< Screen-space coordinates in range [0, clientSize]. (0,0) is the top-left corner of the window.
		Vector2f wheelDelta;      ///< If the current event is CMouseEvent#Type#Wheel, the change in wheel scroll. Otherwise zero.
		InputModifiers mods;    ///< Keyboard modifiers. Only valid if the event Type is one the button events
	};

	struct KeyboardEvent
	{
		/** Keyboard event Type
		*/
		enum class Type
		{
			KeyPressed,     ///< Key was pressed.
			KeyReleased,    ///< Key was released.
			Input           ///< Character input
		};

		/** Use this enum to find out which key was pressed. Alpha-numeric keys use their uppercase ASCII code, so you can use that as well.
		*/
		enum class Key : uint32_t
		{
			// ASCII values. Do not change them.
			Space           = ' ',
			Apostrophe      = '\'',
			Comma           = ',',
			Minus           = '-',
			Period          = '.',
			Slash           = '/',
			Key0            = '0',
			Key1            = '1',
			Key2            = '2',
			Key3            = '3',
			Key4            = '4',
			Key5            = '5',
			Key6            = '6',
			Key7            = '7',
			Key8            = '8',
			Key9            = '9',
			Semicolon       = ';',
			Equal           = '=',
			A               = 'A',
			B               = 'B',
			C               = 'C',
			D               = 'D',
			E               = 'E',
			F               = 'F',
			G               = 'G',
			H               = 'H',
			I               = 'I',
			J               = 'J',
			K               = 'K',
			L               = 'L',
			M               = 'M',
			N               = 'N',
			O               = 'O',
			P               = 'P',
			Q               = 'Q',
			R               = 'R',
			S               = 'S',
			T               = 'T',
			U               = 'U',
			V               = 'V',
			W               = 'W',
			X               = 'X',
			Y               = 'Y',
			Z               = 'Z',
			LeftBracket     = '[',
			Backslash       = '\\',
			RightBracket    = ']',
			GraveAccent     = '`',

			// Special keys
			Escape          ,
			Tab             ,
			Enter           ,
			Backspace       ,
			Insert          ,
			Del             ,
			Right           ,
			Left            ,
			Down            ,
			Up              ,
			PageUp          ,
			PageDown        ,
			Home            ,
			End             ,
			CapsLock        ,
			ScrollLock      ,
			NumLock         ,
			PrintScreen     ,
			Pause           ,
			F1              ,
			F2              ,
			F3              ,
			F4              ,
			F5              ,
			F6              ,
			F7              ,
			F8              ,
			F9              ,
			F10             ,
			F11             ,
			F12             ,
			Keypad0         ,
			Keypad1         ,
			Keypad2         ,
			Keypad3         ,
			Keypad4         ,
			Keypad5         ,
			Keypad6         ,
			Keypad7         ,
			Keypad8         ,
			Keypad9         ,
			KeypadDel       ,
			KeypadDivide    ,
			KeypadMultiply  ,
			KeypadSubtract  ,
			KeypadAdd       ,
			KeypadEnter     ,
			KeypadEqual     ,
			LeftShift       ,
			LeftControl     ,
			LeftAlt         ,
			LeftSuper       , // Windows key on windows
			RightShift      ,
			RightControl    ,
			RightAlt        ,
			RightSuper      , // Windows key on windows
			Menu            ,
		};

		Type type;              ///< The event type
		Key  key;               ///< The last key that was pressed/released
		int glfwKey;			///< The last key in native glfw format
		InputModifiers mods;    ///< Keyboard modifiers
		uint32_t codepoint = 0; ///< UTF-32 codepoint from GLFW for Input event types
	};

	inline KeyboardEvent::Key glfwToKey(int glfwKey)
	{
		static_assert(GLFW_KEY_ESCAPE == 256, "GLFW_KEY_ESCAPE is expected to be 256");
		if (glfwKey < GLFW_KEY_ESCAPE)
		{
			// Printable keys are expected to have the same value
			return (KeyboardEvent::Key)glfwKey;
		}

		switch (glfwKey)
		{
		case GLFW_KEY_ESCAPE:
			return KeyboardEvent::Key::Escape;
		case GLFW_KEY_ENTER:
			return KeyboardEvent::Key::Enter;
		case GLFW_KEY_TAB:
			return KeyboardEvent::Key::Tab;
		case GLFW_KEY_BACKSPACE:
			return KeyboardEvent::Key::Backspace;
		case GLFW_KEY_INSERT:
			return KeyboardEvent::Key::Insert;
		case GLFW_KEY_DELETE:
			return KeyboardEvent::Key::Del;
		case GLFW_KEY_RIGHT:
			return KeyboardEvent::Key::Right;
		case GLFW_KEY_LEFT:
			return KeyboardEvent::Key::Left;
		case GLFW_KEY_DOWN:
			return KeyboardEvent::Key::Down;
		case GLFW_KEY_UP:
			return KeyboardEvent::Key::Up;
		case GLFW_KEY_PAGE_UP:
			return KeyboardEvent::Key::PageUp;
		case GLFW_KEY_PAGE_DOWN:
			return KeyboardEvent::Key::PageDown;
		case GLFW_KEY_HOME:
			return KeyboardEvent::Key::Home;
		case GLFW_KEY_END:
			return KeyboardEvent::Key::End;
		case GLFW_KEY_CAPS_LOCK:
			return KeyboardEvent::Key::CapsLock;
		case GLFW_KEY_SCROLL_LOCK:
			return KeyboardEvent::Key::ScrollLock;
		case GLFW_KEY_NUM_LOCK:
			return KeyboardEvent::Key::NumLock;
		case GLFW_KEY_PRINT_SCREEN:
			return KeyboardEvent::Key::PrintScreen;
		case GLFW_KEY_PAUSE:
			return KeyboardEvent::Key::Pause;
		case GLFW_KEY_F1:
			return KeyboardEvent::Key::F1;
		case GLFW_KEY_F2:
			return KeyboardEvent::Key::F2;
		case GLFW_KEY_F3:
			return KeyboardEvent::Key::F3;
		case GLFW_KEY_F4:
			return KeyboardEvent::Key::F4;
		case GLFW_KEY_F5:
			return KeyboardEvent::Key::F5;
		case GLFW_KEY_F6:
			return KeyboardEvent::Key::F6;
		case GLFW_KEY_F7:
			return KeyboardEvent::Key::F7;
		case GLFW_KEY_F8:
			return KeyboardEvent::Key::F8;
		case GLFW_KEY_F9:
			return KeyboardEvent::Key::F9;
		case GLFW_KEY_F10:
			return KeyboardEvent::Key::F10;
		case GLFW_KEY_F11:
			return KeyboardEvent::Key::F11;
		case GLFW_KEY_F12:
			return KeyboardEvent::Key::F12;
		case GLFW_KEY_KP_0:
			return KeyboardEvent::Key::Keypad0;
		case GLFW_KEY_KP_1:
			return KeyboardEvent::Key::Keypad1;
		case GLFW_KEY_KP_2:
			return KeyboardEvent::Key::Keypad2;
		case GLFW_KEY_KP_3:
			return KeyboardEvent::Key::Keypad3;
		case GLFW_KEY_KP_4:
			return KeyboardEvent::Key::Keypad4;
		case GLFW_KEY_KP_5:
			return KeyboardEvent::Key::Keypad5;
		case GLFW_KEY_KP_6:
			return KeyboardEvent::Key::Keypad6;
		case GLFW_KEY_KP_7:
			return KeyboardEvent::Key::Keypad7;
		case GLFW_KEY_KP_8:
			return KeyboardEvent::Key::Keypad8;
		case GLFW_KEY_KP_9:
			return KeyboardEvent::Key::Keypad9;
		case GLFW_KEY_KP_DECIMAL:
			return KeyboardEvent::Key::KeypadDel;
		case GLFW_KEY_KP_DIVIDE:
			return KeyboardEvent::Key::KeypadDivide;
		case GLFW_KEY_KP_MULTIPLY:
			return KeyboardEvent::Key::KeypadMultiply;
		case GLFW_KEY_KP_SUBTRACT:
			return KeyboardEvent::Key::KeypadSubtract;
		case GLFW_KEY_KP_ADD:
			return KeyboardEvent::Key::KeypadAdd;
		case GLFW_KEY_KP_ENTER:
			return KeyboardEvent::Key::KeypadEnter;
		case GLFW_KEY_KP_EQUAL:
			return KeyboardEvent::Key::KeypadEqual;
		case GLFW_KEY_LEFT_SHIFT:
			return KeyboardEvent::Key::LeftShift;
		case GLFW_KEY_LEFT_CONTROL:
			return KeyboardEvent::Key::LeftControl;
		case GLFW_KEY_LEFT_ALT:
			return KeyboardEvent::Key::LeftAlt;
		case GLFW_KEY_LEFT_SUPER:
			return KeyboardEvent::Key::LeftSuper;
		case GLFW_KEY_RIGHT_SHIFT:
			return KeyboardEvent::Key::RightShift;
		case GLFW_KEY_RIGHT_CONTROL:
			return KeyboardEvent::Key::RightControl;
		case GLFW_KEY_RIGHT_ALT:
			return KeyboardEvent::Key::RightAlt;
		case GLFW_KEY_RIGHT_SUPER:
			return KeyboardEvent::Key::RightSuper;
		case GLFW_KEY_MENU:
			return KeyboardEvent::Key::Menu;
		default:
			KRR_SHOULDNT_GO_HERE;
			return (KeyboardEvent::Key)0;
		}
	}

	class UserInputHandler {
	public:
		virtual bool onKeyEvent(const KeyboardEvent &keyEvent){return false;}
		virtual bool onMouseEvent(const MouseEvent& mouseEvent){return false;}
	};
}

NAMESPACE_END(krr)