#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import subprocess
import os
import PIL
import json
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from collections import deque
from datetime import datetime, timedelta
import time

def definir_puntos():
    puntos = {
        "1": Pose(Point(0.15, 0.00, 0.0), Quaternion(0, 0, 0.9940, 0.1100)),
        "2": Pose(Point(-0.6, 1.98, 0.0), Quaternion(0, 0, -0.7908, 0.6121)),
        "3": Pose(Point(0.03, 2.29, 0.0), Quaternion(0, 0, 0.6460, 0.7633)),
        "4": Pose(Point(1.05, 3.64, 0.0), Quaternion(0, 0, 0.9880, 0.1545)),
        "5": Pose(Point(1.67, 2.76, 0.0), Quaternion(0, 0, -0.0027, 0.9999)),
        "6": Pose(Point(1.45, 0.87, 0.0), Quaternion(0, 0, -0.0887, 0.9961)),
        "7": Pose(Point(1.33, -0.12, 0.0), Quaternion(0, 0, -0.1155, 0.9933)),
        "8": Pose(Point(3.29, 1.49, 0.0), Quaternion(0, 0, -0.7717, 0.6360)),
    }
    return puntos

def llamar_script_gemini(frase_usuario, ruta_imagen=None):
    try:
        script_path = 'extGemini.py'
        argumentos = [
            '/home/alicia/anaconda3/envs/gemini/bin/python',
            script_path,
            frase_usuario
        ]

        # Si hay imagen, usar otro script y añadir ruta
        if ruta_imagen is not None:
            script_path = 'extGeminiImg.py'
            argumentos = [
                '/home/alicia/anaconda3/envs/gemini/bin/python',
                script_path,
                frase_usuario,
                ruta_imagen
            ]

        resultado = subprocess.run(
            argumentos,
            capture_output=True,
            text=True,
            check=True
        )

        salida = resultado.stdout.strip()
        print("Respuesta del script:", salida)
        return salida

    except subprocess.CalledProcessError as e:
        print("Error al ejecutar el script:", e)
        return ""

def enviar_goal(pose, client):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "robot_map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = pose

    rospy.loginfo("Enviando goal...")
    client.send_goal(goal)
    client.wait_for_result()
    result = client.get_result()
    rospy.loginfo("¡Goal alcanzado!")
    return result

def capturar_y_guardar_imagen():
    bridge = CvBridge()
    image_msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=10)
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"/home/alicia/catkin_ws/src/semantic_nav/src/captura_{timestamp}.jpg"
    cv2.imwrite(nombre_archivo, cv_image)
    rospy.loginfo(f"Imagen guardada en {nombre_archivo}")
    return nombre_archivo

def main():
    rospy.init_node('navegador')
    puntos = definir_puntos()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # SENTENCE BERT
    sentencesTea = ["Empty mug", "Kettle on the table","Tea box on the table","Coaster under the mug","Boil water","Make tea","Drink tea","Pour water","Heat water"]
    sentencesTrays = ["Empty trays","Put files in tray","Put envelopes in tray","Take files from tray","Take envelopes from tray"]
    sentencesWork = ["Computer monitor on","Clean keyboard","Functional mouse","Tidy desk","Work on computer","Type on keyboard","Move the mouse","Read the screen"]
    sentencesMed = ["Unopened bandage box","Unused pill blister","Packaged tissue","Tray containing items","Use a bandage","Take medication","Use a tissue"]
    sentencesSnack = ["Desk","Almond Package","Snack package","Eat snacks"]
    sentencesRecycle = ["Partially full paper bin","Empty plastic bag","Throw away paper","Throw away plastic","Throw away trash"]

    embeddingTea = model.encode(sentencesTea)
    embeddingTrays = model.encode(sentencesTrays)
    embeddingWork = model.encode(sentencesWork)
    embeddingMed = model.encode(sentencesMed)
    embeddingSnack = model.encode(sentencesSnack)
    embeddingRecycle = model.encode(sentencesRecycle)

    # MOVE BASE
    client = actionlib.SimpleActionClient('/robot/move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base.")

    while not rospy.is_shutdown():

        # 1: USER INPUT AND DESCRIPTION GENERATION FROM THE COMMAND
        entrada_usuario = input("What do you need?")
        sentencesUser = llamar_script_gemini(f"""You are given a user command expressing a need or desire: {entrada_usuario}

        Reformulate this command into two parts:

        1. A set of descriptions of typical objects that would fulfill this need.
        2. A set of descriptions of the main actions or utilities associated with fulfilling this need.

        All the responses should be in the context of an indoor space corresponding to an office.

        Example:
        User command: "I'm tired"
        - Scene description: "Empty chair", "Unmade bed", "Sharp pencil" (each row should only be one object and the corresponding set of states)
        - Utility description: "Sit down", "Rest on the bed"

        Now, provide the two reformulated descriptions:""")

        embeddingUser = model.encode(sentencesUser)
        
        # 2: REGION SIMILARITY CALCULATION
        cosine_score_tea = model.similarity(embeddingUser, embeddingTea)
        cosine_score_trays = model.similarity(embeddingUser, embeddingTrays)
        cosine_score_work = model.similarity(embeddingUser, embeddingWork)
        cosine_score_med = model.similarity(embeddingUser, embeddingMed)
        cosine_score_snack = model.similarity(embeddingUser, embeddingSnack)
        cosine_score_recycle = model.similarity(embeddingUser, embeddingRecycle)

        scores = {
                    "4": torch.max(cosine_score_tea).item(),
                    "3": torch.max(cosine_score_trays).item(),
                    "2": torch.max(cosine_score_work).item(),
                    "1": torch.max(cosine_score_med).item(),
                    "6": torch.max(cosine_score_snack).item(),
                    "5": torch.max(cosine_score_recycle).item(),
                }
        mejor_punto = max(scores, key=scores.get)
        puntuacion = scores[mejor_punto]

        print(f"\nThe most relevant point is {mejor_punto} (score: {puntuacion:.3f})")
        confirmacion = input(f"Do you want to go to point {mejor_punto}? (y/n): ").strip().lower()

        if confirmacion not in ['y','yes']:
            continue

        # 3: ROBOT NAVIGATION
        enviar_goal(puntos[mejor_punto], client)
        
        # 4: SCENE ANALYSIS
        time.sleep(0.5)
        ruta_imagen = capturar_y_guardar_imagen()
        
        print('4.1-----------------------------')
        # 4.1: CONTEXTUAL SCENE ANALYSIS (BOTTOM-UP)
        posibles_tareas = llamar_script_gemini(f"""
        You are an assistive robot in a domestic environment.

        The robot has captured the attached image.
        
        Step 1: List all objects in the scene with their temporal or contextual state (e.g., "dirty plate", "empty glass", "open drawer", "wet floor").
        
        Step 2: From these elements, extract a list of possible assistive tasks that could be performed **here and now**, and that are semantically grounded (e.g., "clean desk", "close door", "pick up cup").

        Step 3: For each task, justify why it is feasible (based on visible affordances and state). Only include tasks that are visually grounded and executable.

        Only include tasks that meet all of the following criteria:
        - Are **visually grounded**: the task must involve objects clearly visible in the image.
        - Are **contextually relevant**: the task contributes to cleanliness, organization, safety, or fulfills a likely user goal.
        - Are **non-trivial**: avoid purely aesthetic or unnecessary actions (e.g., slightly repositioning objects, adjusting decoration).
        - Are **executable now**, without requiring additional tools or setup.

        Exclude tasks that:
        - Have no clear benefit to the user or environment.
        - Only involve moving items slightly without functional gain.

        Respond in JSON format like this:
        {{
            "objects": ["dirty plate", "open fridge"],
            "tasks": [
                {{"task": "clean plate", "reason": "There is a dirty plate on the table."}},
                {{"task": "close fridge", "reason": "The fridge is visibly open."}}
            ]
        }}
        """, ruta_imagen)

        print('4.2-----------------------------')
        # 4.2: TASK VALIDATION OF USER GOAL (TOP-DOWN)
        validacion_tarea = llamar_script_gemini(f"""
        You are an assistive robot reasoning about a user request based only on a camera image of the environment.

        The user said: "{entrada_usuario}"
        You are now observing the environment through the attached image.

        You must follow these strict rules:

        1. **Do not assume the presence of any object unless it is clearly visible in the image.**
        2. Only refer to **visible** objects, their state, and their spatial or functional relationships.
        3. **Never use general knowledge** about what is usually present in a room. If something is not visible, do not assume it exists.
        4. If a task makes sense **in this type of environment** but cannot be completed due to missing objects or setup, mark it as **"partially_feasible"** and list the required preconditions.
        5. If the task has **no semantic relevance** to what is seen in the scene (e.g., "cook pasta" in an office), mark it as **"not_feasible"**.
        6. If all needed elements are present and no blockers are seen, mark it as **"feasible"**.

        Do not propose creative alternatives. You must only reason from what is observable in the image.

        Return your response in this strict JSON format:

        {{
          "task": "Short sentence indicating the name or purpose of the task"
          "feasibility": "feasible" | "partially_feasible" | "not_feasible",
          "preconditions": [list of missing required objects or actions, if any],
          "reason": "Explanation based strictly on what is visible in the image."
        }}

        Examples:
        - If you see a mug but no liquid source, the task "get a drink" is "partially_feasible".
        - If you see a messy desk and papers, the task "organize desk" is "feasible".
        - If nothing related to drinking or tea is visible, and the request is "make tea", it is "not_feasible".

        Only output valid JSON. Do not include any comments or extra text.
        """, ruta_imagen)

        print('4.3-----------------------------')
        # 4.3: TASK PRIORITIZATION
        tareas_ordenadas = llamar_script_gemini(f"""
        You are a semantic reasoning module for an assistive robot.

        You are given:
        - A list of current tasks from the user (goal-directed): {json.dumps(validacion_tarea)}
        - A list of contextually suggested tasks (emerging from visual analysis): {json.dumps(posibles_tareas)}
        - The current user request is: "{entrada_usuario}"

        Your job is to generate a list of **structured assistive tasks** using **Hierarchical Task Networks (HTN)**, where each **root task** may have one or more **subtasks** that must be performed in a specific logical order, with clear **dependencies**.

        You must strictly follow these rules:

        1. Only use **objects, affordances, and spatial relationships that are directly mentioned in the previous lists**.
        2. **Do not assume the presence** of objects or agents that are not clearly mentioned.
        3. If a subtask requires an object or condition that is missing, **include it explicitly as a deferred subtask**, with a clear reason (e.g., "no water source is present").
        4. **Do not include tasks that contradict each other**. For example, if one task uses an object (e.g. "fill mug"), do not include another that makes that object unavailable (e.g. "put away mug").
        5. Do **not** propose creative workarounds or use general world knowledge.
        6. Remove redundant tasks. Only include semantically distinct tasks.
        7. Within each root task, order subtasks logically, from prerequisites to execution.
        8. Order root tasks from highest priority considering feasibility, functional relevance and potential risk mitigation to less critical or deferred ones.
        9. Only include tasks involving **pick-and-place**, **simple physical interaction** (e.g., push, open, close), or **visual search** that the robot can perform directly.

        Use this JSON structure:

        ```json
        [
          {{
            "root_task": "Short description of main task",
            "status": "ready" | "deferred" | "blocked" | "done",
            "subtasks": [
              {{
                "task": "Subtask description",
                "status": "ready" | "deferred" | "blocked" | "done",
                "reason": "Brief explanation tied to visible evidence",
                "dependencies": ["name of any prior subtask this depends on"]
              }},
              ...
            ]
          }},
          ...
        ]
        """)

        print('4.4-----------------------------')
        # 4.4: TASK REFINEMENT
        tareas_refinadas = llamar_script_gemini(f"""
        You are a task planner verifier for an assistive robot.

        You are given:
        - A proposed hierarchical task network (HTN) plan generated for an assistive robot: {json.dumps(tareas_ordenadas)}
        - The user's request: "{entrada_usuario}"
        - A current visual observation of the environment (attached image).

        Your job is to **refine and verify** the HTN plan. This includes:

        1. Checking that all tasks and subtasks are **fully grounded in the current visible scene**.
        2. **Removing or correcting** any hallucinated steps or unjustified assumptions (e.g., "there is hot water" when no source is visible).
        3. Making sure **no two tasks contradict each other**, such as using and putting away the same object simultaneously.
        4. Ensuring that **all required steps are included**. If a step depends on a missing object, insert a deferred subtask like "Find water source".
        5. Verifying that all dependencies are valid and that subtasks are ordered logically.
        6. Making sure only one semantically correct use of each object is retained (e.g., if a mug is being used to prepare a drink, it shouldn't be put away).
        7. The plan should only include **feasible or logically deferred tasks**. Remove speculative or irrelevant steps.

        **Only return a corrected plan** using this JSON format:

        ```json
        [
          {{
            "root_task": "Short description of main task",
            "status": "ready" | "deferred" | "blocked" | "done",
            "subtasks": [
              {{
                "task": "Subtask description",
                "status": "ready" | "deferred" | "blocked" | "done",
                "reason": "Justification based strictly on visible elements",
                "dependencies": ["task it depends on"]
              }},
              ...
            ]
          }},
          ...
        ]
        Do not include any commentary, explanation, or free text. Only return valid and corrected JSON.
        """, ruta_imagen)

        #print(f"\nTareas sugeridas y priorizadas: {tareas_ordenadas}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
