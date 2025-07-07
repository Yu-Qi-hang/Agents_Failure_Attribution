import os
import re
import json
import random
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm
# --- Helper Functions ---

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _make_api_call(client, model, messages, max_tokens, extra_body=None):
    """Makes an API call to Azure OpenAI."""
    try:
        if extra_body is None:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body = extra_body
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

# --- All-at-Once Method ---

def all_at_once(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, extra_body=None):
    """
    Analyzes chat history by feeding the entire conversation at once to the model.
    """
    print("\n--- Starting All-at-Once Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed for evaluation

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for entry in chat_history
        ])

        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem}\n"
            f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used in prompt
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
        )

        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]

        result = _make_api_call(client, model, messages, max_tokens, extra_body)

        print(f"Prediction for {json_file}:")
        if result:
            print(result)
        else:
            print("Failed to get prediction.")
        print("\n" + "="*50 + "\n")

# --- Step-by-Step Method ---

def step_by_step(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, extra_body=None):
    """
    Analyzes chat history step by step, asking the model at each step if an error occurred.
    """
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed for evaluation

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                f"Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )

            messages=[
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens, extra_body)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}:Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                error_found = True
                break # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                 print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nPrediction for {json_file}:\nNo decisive errors found by step-by-step analysis.")

        print("\n" + "="*50 + "\n")


# --- Binary Search Method ---

def _construct_binary_search_prompt(problem, answer, chat_segment_content, range_description, upper_half_desc, lower_half_desc):
    """Constructs the prompt for the binary search step."""
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user's query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n" # Included as per original code - remove if ground truth shouldn't be used
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
    )

def _report_binary_search_error(chat_history, step, json_file, is_handcrafted):
    """Reports the identified error step from binary search."""
    index_agent = "role" if is_handcrafted else "name"
    entry = chat_history[step]
    agent_name = entry.get(index_agent, 'Unknown Agent')

    print(f"\nPrediction for {json_file}:")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")

def _find_error_in_segment_recursive(client, model: str, max_tokens: int, chat_history: list, problem: str, answer: str, start: int, end: int, json_file: str, is_handcrafted: bool, extra_body):
    """Recursive helper function for binary search analysis."""
    if start > end:
         print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
         _report_binary_search_error(chat_history, end if end >= 0 else 0, json_file, is_handcrafted) # Report something reasonable
         return
    if start == end:
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    index_agent = "role" if is_handcrafted else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Cannot proceed.")
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2 

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc)

    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
        {"role": "user", "content": prompt}
    ]

    print(f"Analyzing step {start}-{end} for {json_file}...")
    result = _make_api_call(client, model, messages, max_tokens, extra_body)

    if not result:
        print(f"API call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
        return

    print(f"LLM Prediction for segment {start}-{end}: {result}")
    result_lower = result.lower() 

    if "upper half" in result_lower:
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted, extra_body)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, extra_body)
    else:
        print(f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}. Randomly choosing a half.")
        if random.randint(0, 1) == 0:
            print("Randomly chose upper half.")
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted, extra_body)
        else:
            print("Randomly chose lower half.")
            new_start = min(mid + 1, end)
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, extra_body)


def binary_search(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, extra_body=None):
    """
    Analyzes chat history using a binary search approach to find the error step.
    """
    print("\n--- Starting Binary Search Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")

        answer = data.get("ground_truth", "") # Keep ground truth if needed for evaluation

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, 0, len(chat_history) - 1, json_file, is_handcrafted, extra_body)



# --- Sliding Window Analysis ---

def _construct_window_prompt(problem, answer, window_content, start_step, end_step):
    """Constructs the prompt for sliding window analysis."""
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation within a specific time window. "
        "Your goal is to determine if any critical error occurred in this segment that could lead to an incorrect solution.\n"
        f"The problem being addressed is: {problem}\n"
        f"The Answer for the problem is: {answer}\n"
        f"Review the conversation segment from step {start_step} to step {end_step}:\n\n{window_content}\n\n"
        "Based on your analysis, please answer the following:\n"
        "1. Does this segment contain a critical error that could derail the solution? (Answer 'Yes' or 'No')\n"
        f"2. If yes, which step number within this window contains the first critical error? (Provide the step number from {start_step} to {end_step})\n"
        "3. Briefly explain the reason for your judgment.\n"
        "Respond ONLY in the format: 1. Yes/No\n2. Step: [number]\n3. Reason: [explanation]"
    )

def sliding_window_json(client, model, max_tokens, extra_body, data, index_agent, json_file):
    window_size=5
    stride=2
    chat_history = data["history"]
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    error_found = False
    total_steps = len(chat_history)
    
    # Process conversation with sliding window
    for start_idx in range(0, max(0, total_steps - window_size + 1), stride):
        end_idx = min(start_idx + window_size - 1, total_steps - 1)
        
        # Extract window segment
        window_segment = chat_history[start_idx:end_idx + 1]
        window_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" 
            for entry in window_segment
        ])
        
        # Construct and send prompt
        prompt = _construct_window_prompt(problem, ground_truth, window_content, start_idx, end_idx)
        messages = [
            {"role": "system", "content": "You are a precise conversation segment analyzer."},
            {"role": "user", "content": prompt}
        ]
        
        response = _make_api_call(client, model, messages, max_tokens, extra_body)
        
        if not response:
            print(f"API call failed for window {start_idx}-{end_idx} in {json_file}")
            continue
            
        print(f"Window {start_idx}-{end_idx} analysis: {response[:100]}...")  # Truncate for readability
        
        # Parse response
        if "1. yes" in response.lower():
            error_found = True
            step_line = next((line for line in response.split('\n') if "2. step:" in line.lower()), None)
            step_num = int(step_line.split(':')[1].strip()) if step_line else start_idx
            if step_num < start_idx:
                step_num = start_idx + int(step_line.split(':')[1].strip())
            agent_name = chat_history[step_num].get(index_agent, 'Unknown Agent')
            print(f"\nPrediction for {json_file}:")
            print(f"Agent Name: {agent_name}")
            print(f"Step Number: {step_num}")
            print(f"Window: {start_idx}-{end_idx}")
            break
            
    if not error_found:
        print(f"\nPrediction for {json_file}:\nNo critical errors found by sliding window analysis.")
    
def sliding_window(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, extra_body=None):
    """
    Analyzes chat history using a sliding window approach to detect error-containing segments.
    """
    print("\n--- Starting Sliding Window Analysis ---\n")

    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        print(f"--- Analyzing File: {json_file} ---")
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        sliding_window_json(client, model, max_tokens, extra_body, data, index_agent, json_file)
        print("\n" + "="*50 + "\n")
        # file_path = os.path.join(directory_path, json_file)
        # data = _load_json_data(file_path)
        # if not data or not data.get("history"):
        #     continue

        # chat_history = data["history"]
        # problem = data.get("question", "")
        # if "ground_truth" in data:
        #     ground_truth = data.get("ground_truth", "")
        # else:
        #     ground_truth = data.get("groundtruth", "")

        # error_found = False
        # total_steps = len(chat_history)
        
        # print(f"--- Analyzing File: {json_file} ---")
        # # Process conversation with sliding window
        # for start_idx in range(0, max(0, total_steps - window_size + 1), stride):
        #     end_idx = min(start_idx + window_size - 1, total_steps - 1)
            
        #     # Extract window segment
        #     window_segment = chat_history[start_idx:end_idx + 1]
        #     window_content = "\n".join([
        #         f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" 
        #         for entry in window_segment
        #     ])
            
        #     # Construct and send prompt
        #     prompt = _construct_window_prompt(problem, ground_truth, window_content, start_idx, end_idx)
        #     messages = [
        #         {"role": "system", "content": "You are a precise conversation segment analyzer."},
        #         {"role": "user", "content": prompt}
        #     ]
            
        #     response = _make_api_call(client, model, messages, max_tokens, extra_body)
            
        #     if not response:
        #         print(f"API call failed for window {start_idx}-{end_idx} in {json_file}")
        #         continue
                
        #     print(f"Window {start_idx}-{end_idx} analysis: {response[:100]}...")  # Truncate for readability
            
        #     # Parse response
        #     if "1. yes" in response.lower():
        #         error_found = True
        #         step_line = next((line for line in response.split('\n') if "2. step:" in line.lower()), None)
        #         step_num = int(step_line.split(':')[1].strip()) if step_line else start_idx
        #         if step_num < start_idx:
        #             step_num = start_idx + int(step_line.split(':')[1].strip())
        #         agent_name = chat_history[step_num].get(index_agent, 'Unknown Agent')
        #         print(f"\nPrediction for {json_file}:")
        #         print(f"Agent Name: {agent_name}")
        #         print(f"Step Number: {step_num}")
        #         print(f"Window: {start_idx}-{end_idx}")
        #         break
                
        # if not error_found:
        #     print(f"\nPrediction for {json_file}:\nNo critical errors found by sliding window analysis.")
        
        # print("\n" + "="*50 + "\n")

# --- Error Propagation Tracing ---

def _build_dependency_graph(chat_history, is_handcrafted):
    """Builds a dependency graph based on semantic content relationships."""
    graph = {}
    index_agent = "role" if is_handcrafted else "name"
    
    # Create initial graph nodes
    for i, entry in enumerate(chat_history):
        agent = entry.get(index_agent, 'Unknown Agent')
        content = entry.get("content", "")
        
        graph[i] = {
            "agent": agent,
            "content": content,
            "dependencies": []
        }
    
    # Detect dependencies based on semantic relationships
    for i in range(1, len(chat_history)):
        current_content = graph[i]["content"].lower()
        
        # Look for references to previous agents or specific content
        for j in range(i):
            prev_agent = graph[j]["agent"]
            prev_content = graph[j]["content"]
            
            # Check if current message explicitly references previous agent
            if prev_agent.lower() in current_content:
                graph[i]["dependencies"].append(j)
                continue
                
            # Check for content keywords that might indicate dependency
            keywords = ["as mentioned", "previously", "earlier", "following up", "as per", "based on"]
            if any(kw in current_content for kw in keywords) and j < i:
                # Check if content topics overlap
                current_topics = set(current_content.split()[:20])  # First 20 words
                prev_topics = set(prev_content.split()[:20])
                if len(current_topics & prev_topics) > 3:  # At least 3 common words
                    graph[i]["dependencies"].append(j)
    
    return graph

def _trace_error_path(graph, start_step):
    """Traces error propagation path from a starting point."""
    path = []
    queue = [start_step]
    visited = set()
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
            
        visited.add(current)
        path.append(current)
        queue.extend(graph[current]["dependencies"])
    
    return path

def _construct_propagation_prompt(problem, answer, step_details):
    """Constructs prompt for error propagation analysis."""
    step_list = "\n".join([
        f"Step {step}: {details['agent']} - {details['content'][:100]}..."
        for step, details in step_details.items()
    ])
    
    return (
        "You are an AI assistant tasked with identifying the root cause of an error in a multi-agent conversation. "
        "The following steps are part of an error propagation path:\n\n"
        f"{step_list}\n\n"
        f"The problem being solved: {problem}\n"
        f"The Answer for the problem is: {answer}\n"
        "Your task is to determine which step contains the original error that started the chain of mistakes. "
        "Please provide:\n"
        "1. The step number of the root error\n"
        "2. The name of the agent responsible\n"
        "3. A brief explanation of why this is the root cause\n"
        "Respond ONLY in the format: Step Number: [number]\nAgent Name: [name]\nReason: [explanation]"
    )

def error_propagation(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, extra_body=None):
    """
    Traces error propagation through agent dependencies using semantic analysis.
    """
    print("\n--- Starting Error Propagation Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    
    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data or not data.get("history"):
            continue
            
        chat_history = data["history"]
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        
        # Step 1: Build dependency graph with semantic relationships
        print(f"--- Analyzing File: {json_file} ---")
        graph = _build_dependency_graph(chat_history, is_handcrafted)
        
        # Visualize the dependency graph (for debugging)
        print(f"Dependency Graph for {json_file}:")
        for i, node in graph.items():
            deps = ", ".join(str(d) for d in node["dependencies"])
            print(f"Step {i} ({node['agent']}): Depends on [{deps}]")
        
        # Step 2: Find initial error using existing method
        print(f"Locating initial error in {json_file}...")
        error_step = None
        
        # Use step-by-step analysis to find the first error
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get("role" if is_handcrafted else "name", 'Unknown Agent')
            content = entry.get('content', '')
            
            prompt = (
                f"Evaluate if this step contains a critical error:\n"
                f"Agent: {agent_name}\nContent: {content}\n\n"
                f"Context: Problem - {problem}\nGround Truth - {ground_truth}\n"
                "Answer ONLY: 'Yes' or 'No'"
            )
            
            messages = [
                {"role": "system", "content": "Error detection expert"},
                {"role": "user", "content": prompt}
            ]
            
            response = _make_api_call(client, model, messages, max_tokens, extra_body)
            
            if response and "yes" in response.lower():
                error_step = idx
                print(f"Initial error detected at step {idx}")
                break
        
        if error_step is None:
            print("No error detected by step-by-step analysis. Using last step as fallback.")
            error_step = len(chat_history) - 1
        
        # Step 3: Trace propagation path
        error_path = _trace_error_path(graph, error_step)
        
        # Step 4: Analyze the propagation path
        path_details = {
            step: {
                "agent": graph[step]["agent"],
                "content": graph[step]["content"]
            } for step in error_path
        }
        
        prompt = _construct_propagation_prompt(problem, ground_truth, path_details)
        messages = [
            {"role": "system", "content": "You are an expert in error root cause analysis."},
            {"role": "user", "content": prompt}
        ]
        
        response = _make_api_call(client, model, messages, max_tokens, extra_body)
        
        print(f"\nPrediction for {json_file}:")
        if response:
            print(response)
        else:
            print(f"Failed to analyze propagation path.")
        
        print("\n" + "="*50 + "\n")

# --- Hybrid Analysis Method ---

def _lightweight_agent_prompt(problem, ground_truth, chat_content, agent_names):
    """轻量级Agent识别提示词"""
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
        # "Identify ONLY the primary responsible agent for the error in this conversation:\n"
        f"Problem: {problem}\n"
        f"Ground Truth: {ground_truth}\n"
        "Conversation:\n" + chat_content + "\n\n"
        f"Optional Agents: {agent_names}\n"
        "Identify which agent made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "Please answer in the format: Agent: (Your prediction)"
    )

def hybrid_analysis(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, extra_body=None):
    """
    混合分析法：先识别责任Agent，再聚焦分析其步骤
    """
    window_size=3
    stride=1
    print("\n--- Starting Hybrid Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"
    
    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
            
        chat_history = data["history"]
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        
        print(f"--- Analyzing File: {json_file} ---")
        # 第一阶段：轻量级Agent识别
        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" 
            for entry in chat_history
        ])
        chat_content = []
        agent_names = []
        for entry in chat_history:
            agent_names.append(f"{entry.get(index_agent, 'Unknown Agent')}")
            chat_content.append(f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}")
        agent_names = list(set(agent_names))
        agent_prompt = _lightweight_agent_prompt(problem, ground_truth, "\n".join(chat_content), ', '.join(agent_names))
        agent_messages = [
            {"role": "system", "content": "You are an AI assistant tasked with identifying the MOST RESPONSIBLE AGENT for the critical error in the conversation."},
            {"role": "user", "content": agent_prompt}
        ]

        print(f"Stage 1: Identifying primary agent for {json_file}...")
        agent_response = _make_api_call(client, model, agent_messages, max_tokens, extra_body)
        
        # 解析Agent响应
        target_agent = None
        if agent_response:
            if "Agent:" in agent_response:
                target_agent = agent_response.split("Agent:")[1].strip()
                print(f"Identified primary agent: {target_agent}")
                if target_agent in agent_names:
                    pass  # 直接使用
                else:
                    matched = [name for name in agent_names if name in target_agent]
                    target_agent = matched[0] if matched else None
        # 第二阶段：聚焦分析
        if target_agent:
            print(f"Stage 2: Focusing on {target_agent}'s actions...")
            # 获取目标Agent的所有步骤位置
            agent_steps = [
                idx for idx, entry in enumerate(chat_history)
                if entry.get(index_agent) == target_agent
            ]
            # 创建聚焦窗口（仅包含目标Agent的步骤）
            windows = []
            for i in range(0, len(agent_steps), stride):
                end_idx = min(i + window_size, len(agent_steps))
                window_indices = agent_steps[i:end_idx]
                
                # 获取实际对话步骤（保留上下文）
                start_ctx = max(0, window_indices[0] - 1)  # 包含前一步上下文
                end_ctx = min(len(chat_history)-1, window_indices[-1] + 1)  # 包含后一步上下文
                
                windows.append({
                    "start": start_ctx,
                    "end": end_ctx,
                    "agent_steps": window_indices
                })
            
            # 扫描窗口
            error_found = False
            for window in windows:
                # 构建窗口内容时明确标注全局步骤编号
                window_content = ""
                for idx in range(window['start'], window['end']+1):
                    entry = chat_history[idx]
                    agent_name = entry.get(index_agent, 'Unknown Agent')
                    content = entry.get('content', '')
                    # 关键修改：在每条消息前添加全局步骤编号
                    window_content += f"[Global Step {idx}] {agent_name}: {content}\n"
                
                # 修改提示词明确要求全局步骤编号
                window_prompt = (
                    f"Analyze steps {window['start']}-{window['end']} (GLOBAL steps) focusing on {target_agent}:\n"
                    f"Problem: {problem}\n"
                    f"Ground Truth: {ground_truth}\n"
                    f"Conversation segment:\n{window_content}\n\n"
                    "Determine:\n"
                    f"1. Does this segment contain the CRITICAL error by this {target_agent}? (Yes/No)\n"
                    "2. If yes, provide the EXACT GLOBAL step number\n"
                    "3. Error type: [Calculation|Logic|Fact|Procedure]\n"
                    "Respond in EXACT format:\n"
                    "Error Present: [Yes/No]\n"
                    "Global Step: [number or None]\n"  # 明确要求全局编号
                    "Error Type: [type or None]"
                )
                
                window_messages = [
                    {"role": "system", "content": f"Error analysis focused on {target_agent}"},
                    {"role": "user", "content": window_prompt}
                ]
                
                print(f"Analyzing steps {window['start']}-{window['end']}...")
                window_response = _make_api_call(client, model, window_messages, max_tokens, extra_body)
                
                if not window_response:
                    print("API call failed for window")
                    continue
                    
                # 解析窗口响应
                error_present = None
                error_step = None
                error_type = None

                for line in window_response.split('\n'):
                    if "Global Step:" in line:
                        try:
                            error_step = int(line.split(":")[1].strip())
                        except:
                            error_step = None
                    elif "Error Present:" in line:
                        error_present = "yes" in line.lower()
                    elif "Error Type:" in line:
                        error_type = line.split(":")[1].strip()

                if error_step is not None:
                    if error_step < window['start'] or error_step > window['end']:
                        print(f"Warning: Step {error_step} outside window {window['start']}-{window['end']}")
                        error_step = None
                    elif chat_history[error_step].get(index_agent) != target_agent:
                        print(f"Warning: Step {error_step} is not from {target_agent}")
                        target_agent = chat_history[error_step].get(index_agent)
                
                if error_present and error_step is not None:
                    # 现在error_step是验证过的全局编号
                    print(f"\nPrediction for {json_file}:")
                    print(f"Agent Name: {target_agent}")
                    print(f"Step Number: {error_step}")
                    print(f"Error Type: {error_type}")
                    error_found = True
                    break

            if not error_found:
                print(f"\nPrediction for {json_file}:")
                # print(f"Agent Name: {target_agent}")
                print(f"No critical error step found in {json_file}")
        else:
            # 回退到标准滑动窗口
            print(f"Unexpected agent response format: {agent_response}. Falling back to sliding window...")
            sliding_window_json(client, model, max_tokens, extra_body, data, index_agent, json_file)
            # sliding_window(client, directory_path, is_handcrafted, model, max_tokens, extra_body=extra_body)
        
        print("\n" + "="*50 + "\n")