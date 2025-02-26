import cv2

def remove_background_opencv(input_video_path, output_video_path, method="MOG"):
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)
    
    # Select background subtractor
    if method == "MOG2":
        fgbg = cv2.createBackgroundSubtractorMOG2()
    elif method == "LSBP":
        fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP()
    else:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply background subtraction
        fgmask = fgbg.apply(frame)
        
        # Ensure the mask is binary (0 or 255)
        _, binary_mask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        
        # Write binary mask frame to output video
        out.write(binary_mask)
        
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("Binary mask video generated. Output saved at:", output_video_path)