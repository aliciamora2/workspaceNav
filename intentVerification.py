import os
import json
import google.generativeai as genai
from ultralytics import YOLO

# --- Configuration ---
os.environ['GOOGLE_API_KEY'] = "KEY_HERE"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel("gemini-3-flash-preview")
yolo_model = YOLO("yolov8m-world.pt")
DB_PATH = 'ontology.json'

def load_ontology():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f: return json.load(f)
    return {}

def save_ontology(data):
    with open(DB_PATH, 'w') as f: json.dump(data, f, indent=4)

def update_ontology_batch(missing_objects, ontology):
    if not missing_objects: return ontology
    print(f" Knowledge Gap: Batch updating ontology for: {missing_objects}...")
    prompt = f"""
    Act as a precise Physical Ontology Expert for a service robot. 
    For these objects: {missing_objects}, return a JSON object.
    STRICT RULES:
    - Affordances (true/false): container, openable, pourable, switchable, movable, surface.
    - compatibility: ["liquid", "solid", "electronic", "none"].
    - primary_category: ["furniture", "kitchenware", "tool", "food_drink", "appliance"].
    Return ONLY JSON.
    """
    response = model.generate_content(prompt)
    try:
        new_data = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        ontology.update(new_data)
        save_ontology(ontology)
    except Exception as e:
        print(f" Error parsing ontology: {e}")
    return ontology

def run_robot_pipeline(image_path):
    ontology = load_ontology()
    user_input = input("Enter command: ")

    # 1. Semantic Extraction
    analysis_prompt = f"""
    Objective: You are a robotic planning unit. Analyze user intent to extract the required objects to satisfy the need: "{user_input}"

    CRITICAL RULES FOR ENTITY SEPARATION:
    1. Do not include target_objects that are unnecesary. Only include those that are critical for acomplishing the need.
    2. Similarity Rule: You may only group objects in 'target_objects' if they are functional substitutes for EACH OTHER (e.g., ["pizza", "snack"] is fine, but ["pizza", "plate"] is NOT).
    3. Minimalist Affordances: Only require affordances strictly necessary for that specific role.
    
    JSON Format:
    - "target_objects": [list of substitute names]
    - "needed_affordances": [container, openable, pourable, switchable, movable, surface]
    - "content_type": "liquid" | "solid" | "electronic" | "none"
    """
    res = model.generate_content(analysis_prompt)
    try:
        required_entities = json.loads(res.text.replace("```json", "").replace("```", "").strip())
        if isinstance(required_entities, dict): required_entities = [required_entities]
    except Exception as e:
        print(f" Failed to parse JSON: {e}")
        return

    print("\nROBOT INTENT (MLLM Reasoning):")
    for idx, ent in enumerate(required_entities):
        print(f"  Entity {idx+1}: Targets: {ent.get('target_objects')} | Needs: {ent.get('needed_affordances')} | Type: {ent.get('content_type')}")

    # 2. Vision System: GLOBAL FUSION (A + B)
    print("\n---  Vision System: Exhaustive Fusion ---")
    
    # A. Bottom-Up (General context)
    results_bu = yolo_model.predict(image_path, conf=0.2, verbose=False)
    found_bu = [yolo_model.names[int(c)] for c in results_bu[0].boxes.cls]
    print(f"Phase A (Bottom-Up) Detections: {list(set(found_bu))}")
    results_bu[0].show()

    # B. Top-Down (Targeted search)
    all_potential_labels = [name for ent in required_entities for name in ent.get('target_objects', [])]
    yolo_model.set_classes(all_potential_labels)
    results_td = yolo_model.predict(image_path, conf=0.3, verbose=False)
    found_td = [yolo_model.names[int(c)] for c in results_td[0].boxes.cls]
    print(f"Phase B (Top-Down) Detections: {list(set(found_td))}")
    results_td[0].show()

    # Fusion for Reasoning
    all_detections = list(set(found_bu + found_td))
    print(f"Total Combined Unique Detections: {all_detections}")

    # 3. Update Ontology
    ontology = update_ontology_batch([o for o in all_detections + all_potential_labels if o not in ontology], ontology)

    # 4. Hierarchical Reasoning Log (FULL VERBOSE LOGS)
    print("\n" + "="*40)
    print("ROBOT REASONING LOG (Exhaustive Fusion)")
    print("="*40)
    
    satisfied_entities = {}
    
    for i, entity in enumerate(required_entities):
        target_list = [t.lower() for t in entity.get('target_objects', [])]
        target_content = entity.get('content_type', 'none')
        print(f"\nEvaluating Requirement {i+1} ({' / '.join(target_list)}):")
        
        valid_options = []

        for obj in all_detections:
            props = ontology.get(obj, {})
            obj_lower = obj.lower()
            obj_content = props.get('compatibility', 'none')
            
            # Condition 1: Direct Name Match
            name_match = any(t in obj_lower or obj_lower in t for t in target_list)
            
            # Condition 2: Utility Match
            needed = entity.get('needed_affordances', [])
            match_aff = all(props.get(a, False) for a in needed)
            # 'none' bypass for content compatibility
            match_cont = (target_content == "none" or obj_content == target_content)

            print(f"  > Checking '{obj}': ", end="")
            if (name_match or match_aff) and match_cont:
                status = "IDEAL" if name_match else "SUBSTITUTE"
                print(f" VALID ({status})")
                valid_options.append(obj)
            else:
                reasons = []
                if not name_match and not match_aff: reasons.append("affordance mismatch")
                if not match_cont: reasons.append(f"content mismatch ({obj_content} vs {target_content})")
                print(f"REJECTED ({', '.join(reasons)})")

        satisfied_entities[f"Req_{i+1}"] = list(set(valid_options))

    # 5. Final Verdict
    print("\n" + "="*40)
    print(" FINAL FEASIBILITY SUMMARY")
    print("="*40)
    
    all_satisfied = True
    for req_name, options in satisfied_entities.items():
        if not options:
            print(f"{req_name}: FAILED")
            all_satisfied = False
        elif len(options) == 1:
            print(f"{req_name}: SUCCESS using {options[0]}")
        else:
            print(f"{req_name}: AMBIGUITY! I found {options}. Which one do you prefer?")
            all_satisfied = False 

    print(f"\nCONCLUSION: {'TASK FEASIBLE' if all_satisfied else 'TASK INCOMPLETE'}")

if __name__ == "__main__":
    run_robot_pipeline("imgs/mesaComida.png")
