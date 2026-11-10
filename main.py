from scene_engine import SceneEngine
from image_engine import ImageEngine
from video_engine import VideoEngine

def run_pipeline(script_text: str):
    print("=== Starting Cinematic AI Content Pipeline ===")
    
    # Step 1: Script to Scenes breakdown
    scene_engine = SceneEngine()
    scenes = scene_engine.parse_script_to_scenes(script_text)
    print(f"-> Generated {len(scenes)} scenes from script.")
    
    # Step 2: Image Engine & Character Moments Generation
    image_engine = ImageEngine()
    visual_moments = image_engine.generate_character_moments(scenes)
    print("-> Visual moments and character prompts generated successfully.")
    
    # Step 3: Video Engine Assembly
    video_engine = VideoEngine()
    final_output = video_engine.assemble_video(visual_moments)
    
    print(f"=== Pipeline Completed! Final Video Ready at: {final_output} ===")
    return final_output

if __name__ == "__main__":
    sample_script = """
    Zindagi ke safar mein akela chal raha tha wo.
    Tabhi achanak ek nayi roshni ne uska rasta roshan kar diya.
    Usne apne sapno ko sach karne ki thaan li.
    """
    run_pipeline(sample_script)
