import os
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()

from src.agent.graph import create_graph

if "GOOGLE_API_KEY" in os.environ:
    print(f"GOOGLE_API_KEY loaded: {os.environ['GOOGLE_API_KEY'][:5]}...")
else:
    print("GOOGLE_API_KEY NOT FOUND in environment")


def main():
    print("Starting Data Analysis Agent...")
    
    # Initialize the graph
    app = create_graph()
    
    # Initial State
    initial_state = {
        "file_path": "src/data/sample.csv",
        "steps_log": [],
        "analysis_results": []
    }
    
    # Run the graph
    # We use stream to see steps
    thread = {"configurable": {"thread_id": "1"}}
    
    print(f"Processing {initial_state['file_path']}...")
    
    for event in app.stream(initial_state, thread):
        for key, value in event.items():
            print(f"\n--- Node: {key} ---")
            if "steps_log" in value:
                print(f"Log: {value['steps_log'][-1]}")
            if "evaluation_feedback" in value:
                print(f"Decision: {value['evaluation_feedback']}")
            if "final_report" in value:
                print(f"\n[Final Report Generated]")
                # print(value['final_report'][:200] + "...") # Preview
            
    # Handle Interrupt (Human Review)
    snapshot = app.get_state(thread)
    if snapshot.next:
        print("\n--- Human Review Required ---")
        print("The agent has generated a report. Do you approve? (yes/no/[feedback])")
        
        # Simulating user input for the demo. 
        # In real usage, you would use input()
        # user_input = input("Your response: ")
        
        # For autonomous verification, we will inject a rejection once, then approve.
        # But to keep it simple, let's just APPROVE.
        user_input = "APPROVE" 
        print(f"User Input (Simulated): {user_input}")
        
        if "yes" in user_input.lower() or "approve" in user_input.upper():
            app.update_state(thread, {"human_feedback": "APPROVE"})
        else:
            app.update_state(thread, {"human_feedback": f"REJECT: {user_input}"})
            
        print("Resuming graph...")
        for event in app.stream(None, thread):
            for key, value in event.items():
                 print(f"\n--- Node: {key} ---")

    print("\nWorkflow Completed.")

if __name__ == "__main__":
    main()
