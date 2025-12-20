#!/usr/bin/env python3
"""
Demo program demonstrating Ollama function calling with a simple Python tool.

This program shows how to:
1. Define a Python function as a tool (shell/template for actual functionality)
2. Use Ollama with a tool-supporting model to call the tool
3. Handle tool calls and responses

The tool function is a shell that can be replaced with actual functionality like:
- Bluetooth lighting control
- Sound synthesis/playback
- Robot arm control
- Or any other hardware/software control

Usage:
    python test-mcp.py

Requirements:
    - ollama package: pip install ollama
    - Ollama server running: ollama serve
    - A model that supports tools pulled:
      * ollama pull functiongemma (recommended - 270M, designed for function calling)
      * ollama pull gemma3nTools:2b (or gemma3nTools:4b)
      * ollama pull llama3.1 (also supports tools)
      * ollama pull mistral-nemo (also supports tools)

Note: Regular gemma3n models do NOT support tools. Use functiongemma or gemma3nTools instead.
"""

import json
import sys
from typing import Optional


def device_control(action: str, device_id: Optional[str] = None, value: Optional[float] = None) -> str:
    """
    Control a device with various actions.
    
    This is a shell function that can be replaced with actual device control logic.
    Examples: Bluetooth lighting, sound synthesis/playback, robot arm movement, etc.
    
    Args:
        action: The action to perform. Can be 'on', 'off', 'set', 'status', or 'reset'.
        device_id: Optional device identifier. If None, uses the default device.
        value: Optional numeric value for 'set' action (e.g., brightness 0-100, position, volume).
    
    Returns:
        str: Status message indicating the result of the operation.
    """
    try:
        # TODO: Replace this shell with actual device control implementation
        # Example for Bluetooth lighting:
        #   from bleak import BleakClient
        #   async with BleakClient(device_address) as client:
        #       await client.write_gatt_char(characteristic_uuid, command)
        #
        # Example for sound synthesis:
        #   import pygame
        #   pygame.mixer.init()
        #   sound = pygame.mixer.Sound(file_path)
        #   sound.play()
        #
        # Example for robot arm:
        #   import serial
        #   ser = serial.Serial(port, baudrate)
        #   ser.write(f"MOV {value}".encode())
        
        device_str = f" '{device_id}'" if device_id else " (default)"
        
        if action == "on":
            result = f"Device{device_str} turned ON successfully."
        elif action == "off":
            result = f"Device{device_str} turned OFF successfully."
        elif action == "set":
            if value is None:
                result = f"Error: 'set' action requires a value parameter."
            else:
                result = f"Device{device_str} set to {value} successfully."
        elif action == "status":
            result = f"Device{device_str} status: ACTIVE, value=50.0 (example)"
        elif action == "reset":
            result = f"Device{device_str} reset to default settings."
        else:
            result = f"Error: Unknown action '{action}'. Supported actions: on, off, set, status, reset."
        
        print(f"[Device Control] {result}", file=sys.stderr)
        return result
        
    except Exception as e:
        error_msg = f"Failed to control device{device_str}: {str(e)}"
        print(f"[Device Control Error] {error_msg}", file=sys.stderr)
        return error_msg


def main():
    """Main function to demonstrate tool calling with Ollama."""
    try:
        import ollama
    except ImportError:
        print("Error: ollama package not found. Install it with: pip install ollama", file=sys.stderr)
        sys.exit(1)
    
    # Check if Ollama server is accessible
    try:
        client = ollama.Client()
        # Try to list models to verify connection
        models = client.list()
        print("Connected to Ollama server.", file=sys.stderr)
    except Exception as e:
        print(f"Error: Could not connect to Ollama server. Make sure 'ollama serve' is running.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Models that support function calling (in order of preference)
    # Note: Regular gemma3n models do NOT support tools!
    tool_supporting_models = [
        "functiongemma",           # 270M, designed specifically for function calling
        "gemma3nTools:2b",         # 2B variant with tool support
        "gemma3nTools:4b",         # 4B variant with tool support
        "gemma3nTools",            # Base gemma3nTools
        "llama3.1",                # Also supports tools
        "mistral-nemo",            # Also supports tools
    ]
    
    model_name = None
    try:
        models = client.list()
        model_names = [model['name'] for model in models.get('models', [])]
        
        # Try to find a model that supports tools
        for pref_model in tool_supporting_models:
            # Check for exact match or prefix match
            for name in model_names:
                if name == pref_model or name.startswith(pref_model + ":"):
                    model_name = name
                    break
            if model_name:
                break
        
        if not model_name:
            print(f"Warning: No tool-supporting model found. Available models: {model_names}", file=sys.stderr)
            print(f"\nTo use this demo, pull a model that supports tools:", file=sys.stderr)
            print(f"  ollama pull functiongemma          # Recommended (270M, designed for function calling)", file=sys.stderr)
            print(f"  ollama pull gemma3nTools:2b         # 2B variant", file=sys.stderr)
            print(f"  ollama pull gemma3nTools:4b         # 4B variant", file=sys.stderr)
            print(f"  ollama pull llama3.1               # Also supports tools", file=sys.stderr)
            print(f"\nNote: Regular gemma3n models do NOT support tools!", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using model: {model_name}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not list models: {e}", file=sys.stderr)
        print("Using default: functiongemma", file=sys.stderr)
        model_name = "functiongemma"
    
    # Initialize conversation messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can control devices. When the user asks you to control a device, use the device_control tool."
        }
    ]
    
    # Example user queries to demonstrate tool calling
    example_queries = [
        "Turn on the device",
        "What's the status of the device?",
        "Set the device to 75",
        "Turn off the device",
    ]
    
    print("\n" + "="*70, file=sys.stderr)
    print("Ollama Tool Calling Demo", file=sys.stderr)
    print("="*70 + "\n", file=sys.stderr)
    print(f"Using model: {model_name}", file=sys.stderr)
    print("This demo will run through example queries that trigger tool calls.\n", file=sys.stderr)
    
    for i, user_query in enumerate(example_queries, 1):
        print(f"\n--- Example {i}: {user_query} ---", file=sys.stderr)
        print(f"\nUser: {user_query}")
        
        # Add user message
        messages.append({"role": "user", "content": user_query})
        
        # Call Ollama with the tool
        try:
            response = client.chat(
                model=model_name,
                messages=messages,
                tools=[device_control],  # Pass the function directly - Ollama will convert it
                stream=False,
            )
        except Exception as e:
            print(f"Error calling Ollama: {e}", file=sys.stderr)
            continue
        
        # Check if the model wants to call a tool
        # Handle both dict and object access patterns
        if hasattr(response, 'message'):
            message = response.message
            if hasattr(message, 'tool_calls'):
                tool_calls = message.tool_calls or []
            else:
                tool_calls = []
        else:
            message = response.get('message', {})
            tool_calls = message.get('tool_calls', [])
        
        if tool_calls:
            print("\n[Model decided to call tool(s)]\n")
            for tool_call in tool_calls:
                # Handle both dict and object access
                if hasattr(tool_call, 'function'):
                    func_obj = tool_call.function
                    func_name = getattr(func_obj, 'name', '')
                    func_args = getattr(func_obj, 'arguments', {})
                else:
                    func_info = tool_call.get('function', {})
                    func_name = func_info.get('name', '')
                    func_args_raw = func_info.get('arguments', {})
                    # Arguments might be a string (JSON) or dict
                    if isinstance(func_args_raw, str):
                        try:
                            func_args = json.loads(func_args_raw)
                        except json.JSONDecodeError:
                            func_args = {}
                    else:
                        func_args = func_args_raw or {}
                
                print(f"Tool call: {func_name}")
                print(f"Arguments: {json.dumps(func_args, indent=2)}\n")
                
                # Execute the tool
                if func_name == 'device_control':
                    tool_result = device_control(**func_args)
                else:
                    tool_result = f"Unknown tool: {func_name}"
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "name": func_name,
                    "content": tool_result,
                })
                
                print(f"Tool result: {tool_result}\n")
            
            # Get final response from model after tool execution
            try:
                final_response = client.chat(
                    model=model_name,
                    messages=messages,
                    stream=False,
                )
                # Handle both dict and object access
                if hasattr(final_response, 'message'):
                    final_message = final_response.message
                    final_content = getattr(final_message, 'content', '') or ''
                else:
                    final_message = final_response.get('message', {})
                    final_content = final_message.get('content', '') or ''
                print(f"Assistant: {final_content}\n")
                
                # Add assistant response to messages
                messages.append({
                    "role": "assistant",
                    "content": final_content,
                })
            except Exception as e:
                print(f"Error getting final response: {e}", file=sys.stderr)
        else:
            # No tool call, just regular response
            if hasattr(message, 'content'):
                content = getattr(message, 'content', '') or ''
            else:
                content = message.get('content', '') or ''
            print(f"\nAssistant: {content}\n")
            
            # Add assistant response to messages
            messages.append({
                "role": "assistant",
                "content": content,
            })
    
    print("\n" + "="*70, file=sys.stderr)
    print("Demo complete!", file=sys.stderr)
    print("="*70 + "\n", file=sys.stderr)
    
    # Interactive mode
    print("\nEntering interactive mode. Type your queries (or 'quit' to exit):\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Call Ollama with the tool
            try:
                response = client.chat(
                    model=model_name,
                    messages=messages,
                    tools=[device_control],
                    stream=False,
                )
            except Exception as e:
                print(f"Error calling Ollama: {e}", file=sys.stderr)
                continue
            
            # Handle response
            # Handle both dict and object access patterns
            if hasattr(response, 'message'):
                message = response.message
                if hasattr(message, 'tool_calls'):
                    tool_calls = message.tool_calls or []
                else:
                    tool_calls = []
            else:
                message = response.get('message', {})
                tool_calls = message.get('tool_calls', [])
            
            if tool_calls:
                # Handle tool calls
                for tool_call in tool_calls:
                    # Handle both dict and object access
                    if hasattr(tool_call, 'function'):
                        func_obj = tool_call.function
                        func_name = getattr(func_obj, 'name', '')
                        func_args = getattr(func_obj, 'arguments', {})
                    else:
                        func_info = tool_call.get('function', {})
                        func_name = func_info.get('name', '')
                        func_args_raw = func_info.get('arguments', {})
                        # Arguments might be a string (JSON) or dict
                        if isinstance(func_args_raw, str):
                            try:
                                func_args = json.loads(func_args_raw)
                            except json.JSONDecodeError:
                                func_args = {}
                        else:
                            func_args = func_args_raw or {}
                    
                    if func_name == 'device_control':
                        tool_result = device_control(**func_args)
                    else:
                        tool_result = f"Unknown tool: {func_name}"
                    
                    messages.append({
                        "role": "tool",
                        "name": func_name,
                        "content": tool_result,
                    })
                
                # Get final response
                try:
                    final_response = client.chat(
                        model=model_name,
                        messages=messages,
                        stream=False,
                    )
                    # Handle both dict and object access
                    if hasattr(final_response, 'message'):
                        final_message = final_response.message
                        final_content = getattr(final_message, 'content', '') or ''
                    else:
                        final_message = final_response.get('message', {})
                        final_content = final_message.get('content', '') or ''
                    print(f"\nAssistant: {final_content}\n")
                    
                    messages.append({
                        "role": "assistant",
                        "content": final_content,
                    })
                except Exception as e:
                    print(f"Error getting final response: {e}", file=sys.stderr)
            else:
                # No tool call, just regular response
                if hasattr(message, 'content'):
                    content = getattr(message, 'content', '') or ''
                else:
                    content = message.get('content', '') or ''
                print(f"\nAssistant: {content}\n")
                
                messages.append({
                    "role": "assistant",
                    "content": content,
                })
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
