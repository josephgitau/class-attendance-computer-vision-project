import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from datetime import datetime, date
import json
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import shutil
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Import ultralytics with warning suppression
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

from ultralytics import YOLO

# Configure Streamlit page
st.set_page_config(
    page_title="Student Attendance System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = None
if 'students_list' not in st.session_state:
    st.session_state.students_list = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Configuration
ATTENDANCE_CSV = "attendance_records.csv"
STUDENTS_CSV = "students_list.csv"
ENROLLED_FACES_DIR = "enrolled_faces"
MODEL_PATH = "runs/detect/student_face_detection/weights/best.pt"

# Create necessary directories
os.makedirs(ENROLLED_FACES_DIR, exist_ok=True)

def show_messages():
    """Display and clear messages from session state"""
    if st.session_state.messages:
        for msg_type, message in st.session_state.messages:
            if msg_type == "success":
                st.success(message)
            elif msg_type == "warning":
                st.warning(message)
            elif msg_type == "error":
                st.error(message)
            elif msg_type == "info":
                st.info(message)
        st.session_state.messages = []  # Clear messages after displaying

def add_message(msg_type, message):
    """Add a message to session state"""
    st.session_state.messages.append((msg_type, message))

def load_model():
    """Load the trained YOLO model"""
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            st.success("✅ Trained model loaded successfully!")
            return model
        else:
            # Fallback to pretrained model
            model = YOLO('yolov8n.pt')
            st.warning("⚠️ Using pretrained model. Train on your data first!")
            return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        # Try to load without GPU
        try:
            import torch
            if torch.cuda.is_available():
                model = YOLO('yolov8n.pt')
                model.to('cpu')  # Force CPU usage
                st.warning("⚠️ Loaded model on CPU due to GPU issues")
                return model
        except:
            pass
        return None

def load_attendance_data():
    """Load attendance data from CSV"""
    try:
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            # Create empty dataframe with required columns
            df = pd.DataFrame(columns=['student_name', 'date', 'time', 'timestamp', 'confidence', 'method'])
            return df
    except Exception as e:
        st.error(f"Error loading attendance data: {str(e)}")
        return pd.DataFrame(columns=['student_name', 'date', 'time', 'timestamp', 'confidence', 'method'])

def load_students_list():
    """Load students list from CSV"""
    try:
        if os.path.exists(STUDENTS_CSV):
            df = pd.read_csv(STUDENTS_CSV)
            return df['student_name'].tolist()
        else:
            return []
    except Exception as e:
        st.error(f"Error loading students list: {str(e)}")
        return []

def save_attendance_record(student_name, confidence, method):
    """Save attendance record to CSV - only once per student per day"""
    try:
        now = datetime.now()
        today_date = now.strftime('%Y-%m-%d')
        
        # Load existing data
        df = load_attendance_data()
        
        # Check if student already marked present today
        if not df.empty:
            today_records = df[(df['student_name'] == student_name) & (df['date'] == today_date)]
            if not today_records.empty:
                return "already_marked"
        
        # Create new record
        record = {
            'student_name': student_name,
            'date': today_date,
            'time': now.strftime('%H:%M:%S'),
            'timestamp': now,
            'confidence': confidence,
            'method': method
        }
        
        # Add new record
        new_df = pd.DataFrame([record])
        if df.empty:
            df = new_df
        else:
            df = pd.concat([df, new_df], ignore_index=True)
        
        # Save to CSV
        df.to_csv(ATTENDANCE_CSV, index=False)
        st.session_state.attendance_df = df
        
        return "success"
    except Exception as e:
        st.error(f"Error saving attendance record: {str(e)}")
        return "error"

def detect_faces(image, model, conf_threshold=0.5):
    """Detect faces in image using YOLO model"""
    try:
        results = model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False
        )
        
        detected_students = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = model.names[cls]
                    
                    detected_students.append({
                        'name': class_name,
                        'confidence': conf,
                        'box': box.xyxy[0].cpu().numpy()
                    })
        
        return detected_students, results[0] if results else None
    except Exception as e:
        st.error(f"Error detecting faces: {str(e)}")
        return [], None

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    img_array = np.array(image)
    
    if detections:
        for detection in detections:
            x1, y1, x2, y2 = detection['box'].astype(int)
            name = detection['name']
            conf = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{name} ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_array, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img_array, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return Image.fromarray(img_array)

# Sidebar navigation
st.sidebar.title("📚 Student Attendance System")

# Check if page was selected from home buttons
if 'selected_page' in st.session_state:
    page = st.session_state.selected_page
    del st.session_state.selected_page  # Clear it after use
else:
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "✅ Mark Attendance", "👥 Enroll Student", "📊 Records & Statistics", "⚙️ Settings"]
    )

# Load model and data
if st.session_state.model is None:
    st.session_state.model = load_model()

if st.session_state.attendance_df is None:
    st.session_state.attendance_df = load_attendance_data()

if not st.session_state.students_list:
    st.session_state.students_list = load_students_list()

# Main content based on selected page
if page == "🏠 Home":
    st.title("🏠 Student Attendance System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Students", len(st.session_state.students_list))
    
    with col2:
        today_attendance = len(st.session_state.attendance_df[
            st.session_state.attendance_df['date'] == date.today().strftime('%Y-%m-%d')
        ]) if not st.session_state.attendance_df.empty else 0
        st.metric("Today's Attendance", today_attendance)
    
    with col3:
        total_records = len(st.session_state.attendance_df) if not st.session_state.attendance_df.empty else 0
        st.metric("Total Records", total_records)
    
    st.markdown("### 🎯 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Mark Attendance", use_container_width=True):
            st.session_state.selected_page = "✅ Mark Attendance"
            st.rerun()
    
    with col2:
        if st.button("👥 Enroll New Student", use_container_width=True):
            st.session_state.selected_page = "👥 Enroll Student"
            st.rerun()
    
    with col3:
        if st.button("📊 View Records", use_container_width=True):
            st.session_state.selected_page = "📊 Records & Statistics"
            st.rerun()

elif page == "✅ Mark Attendance":
    st.title("✅ Mark Attendance")
    st.markdown("---")
    
    # Show any pending messages
    show_messages()
    
    if st.session_state.model is None:
        st.error("❌ Model not loaded. Please check the model path in settings.")
        st.stop()
    
    # Attendance method selection
    method = st.radio(
        "Choose attendance method:",
        ["📷 Camera Capture", "🖼️ Upload Image"],
        horizontal=True
    )
    
    if method == "📷 Camera Capture":
        st.subheader("📷 Camera Capture")
        
        # Camera input
        camera_image = st.camera_input("Take a picture for attendance")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Captured Image", use_container_width=True)
            
            with col2:
                st.subheader("🔍 Detection Results")
                
                with st.spinner("Detecting faces..."):
                    detections, result = detect_faces(image, st.session_state.model)
                
                if detections:
                    annotated_image = draw_detections(image, detections)
                    st.image(annotated_image, caption="Detected Faces", use_container_width=True)
                    
                    st.success(f"✅ Detected {len(detections)} face(s)")
                    
                    for i, detection in enumerate(detections):
                        student_name = detection['name']
                        confidence = detection['confidence']
                        
                        st.write(f"**{i+1}. {student_name}** (Confidence: {confidence:.2%})")
                        
                        if st.button(f"Mark {student_name} Present", key=f"mark_{i}"):
                            result = save_attendance_record(student_name, confidence, "Camera")
                            if result == "success":
                                add_message("success", f"✅ {student_name} marked present!")
                                st.rerun()
                            elif result == "already_marked":
                                add_message("warning", f"⚠️ {student_name} already marked present today!")
                            else:
                                add_message("error", f"❌ Failed to mark {student_name} present!")
                else:
                    st.warning("⚠️ No faces detected. Please try again.")
    
    else:  # Upload Image
        st.subheader("🖼️ Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing student faces"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.subheader("🔍 Detection Results")
                
                with st.spinner("Detecting faces..."):
                    detections, result = detect_faces(image, st.session_state.model)
                
                if detections:
                    annotated_image = draw_detections(image, detections)
                    st.image(annotated_image, caption="Detected Faces", use_container_width=True)
                    
                    st.success(f"✅ Detected {len(detections)} face(s)")
                    
                    for i, detection in enumerate(detections):
                        student_name = detection['name']
                        confidence = detection['confidence']
                        
                        st.write(f"**{i+1}. {student_name}** (Confidence: {confidence:.2%})")
                        
                        if st.button(f"Mark {student_name} Present", key=f"mark_upload_{i}"):
                            result = save_attendance_record(student_name, confidence, "Upload")
                            if result == "success":
                                add_message("success", f"✅ {student_name} marked present!")
                                st.rerun()
                            elif result == "already_marked":
                                add_message("warning", f"⚠️ {student_name} already marked present today!")
                            else:
                                add_message("error", f"❌ Failed to mark {student_name} present!")
                else:
                    st.warning("⚠️ No faces detected. Please try again.")

elif page == "👥 Enroll Student":
    st.title("👥 Enroll New Student")
    st.markdown("---")
    
    # Show any pending messages
    show_messages()
    
    st.subheader("📝 Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        student_name = st.text_input(
            "Student Name",
            placeholder="Enter student's full name",
            help="Enter the student's name exactly as it should appear in attendance records"
        )
        
        student_id = st.text_input(
            "Student ID (Optional)",
            placeholder="Enter student ID",
            help="Optional student identification number"
        )
    
    with col2:
        class_name = st.text_input(
            "Class/Grade",
            placeholder="e.g., Grade 10A",
            help="Student's class or grade"
        )
        
        enrollment_method = st.selectbox(
            "Enrollment Method",
            ["📷 Camera Capture", "🖼️ Upload Images"]
        )
    
    if student_name:
        st.subheader("📸 Capture Student Photos")
        st.info("💡 Tip: Capture multiple photos from different angles for better recognition accuracy")
        
        if enrollment_method == "📷 Camera Capture":
            num_photos = st.slider("Number of photos to capture", 1, 10, 5)
            
            captured_images = []
            
            for i in range(num_photos):
                st.write(f"**Photo {i+1}/{num_photos}**")
                camera_image = st.camera_input(f"Take photo {i+1}", key=f"camera_{i}")
                
                if camera_image is not None:
                    image = Image.open(camera_image)
                    captured_images.append(image)
                    st.image(image, width=200)
            
            if len(captured_images) > 0:
                if st.button("💾 Enroll Student", type="primary"):
                    # Save images and update student list
                    student_dir = os.path.join(ENROLLED_FACES_DIR, student_name.replace(" ", "_"))
                    os.makedirs(student_dir, exist_ok=True)
                    
                    for i, img in enumerate(captured_images):
                        img_path = os.path.join(student_dir, f"{student_name.replace(' ', '_')}_{i+1}.jpg")
                        img.save(img_path)
                    
                    # Update students list
                    students_df = pd.DataFrame({
                        'student_name': [student_name],
                        'student_id': [student_id if student_id else ""],
                        'class': [class_name if class_name else ""],
                        'enrollment_date': [datetime.now().strftime('%Y-%m-%d')],
                        'num_photos': [len(captured_images)]
                    })
                    
                    # Save to CSV
                    if os.path.exists(STUDENTS_CSV):
                        existing_df = pd.read_csv(STUDENTS_CSV)
                        students_df = pd.concat([existing_df, students_df], ignore_index=True)
                    
                    students_df.to_csv(STUDENTS_CSV, index=False)
                    st.session_state.students_list = load_students_list()
                    
                    st.success(f"✅ {student_name} enrolled successfully with {len(captured_images)} photos!")
                    st.balloons()
        
        else:  # Upload Images
            uploaded_files = st.file_uploader(
                "Upload student photos",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Upload multiple photos of the student for better recognition"
            )
            
            if uploaded_files:
                st.write(f"**Selected {len(uploaded_files)} photos:**")
                
                cols = st.columns(min(len(uploaded_files), 5))
                images = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    images.append(image)
                    
                    with cols[i % 5]:
                        st.image(image, width=150)
                
                if st.button("💾 Enroll Student", type="primary"):
                    # Save images and update student list
                    student_dir = os.path.join(ENROLLED_FACES_DIR, student_name.replace(" ", "_"))
                    os.makedirs(student_dir, exist_ok=True)
                    
                    for i, img in enumerate(images):
                        img_path = os.path.join(student_dir, f"{student_name.replace(' ', '_')}_{i+1}.jpg")
                        img.save(img_path)
                    
                    # Update students list
                    students_df = pd.DataFrame({
                        'student_name': [student_name],
                        'student_id': [student_id if student_id else ""],
                        'class': [class_name if class_name else ""],
                        'enrollment_date': [datetime.now().strftime('%Y-%m-%d')],
                        'num_photos': [len(images)]
                    })
                    
                    # Save to CSV
                    if os.path.exists(STUDENTS_CSV):
                        existing_df = pd.read_csv(STUDENTS_CSV)
                        students_df = pd.concat([existing_df, students_df], ignore_index=True)
                    
                    students_df.to_csv(STUDENTS_CSV, index=False)
                    st.session_state.students_list = load_students_list()
                    
                    st.success(f"✅ {student_name} enrolled successfully with {len(images)} photos!")
                    st.balloons()
    
    st.markdown("---")
    st.subheader("📋 Currently Enrolled Students")
    
    if st.session_state.students_list:
        if os.path.exists(STUDENTS_CSV):
            students_df = pd.read_csv(STUDENTS_CSV)
            st.dataframe(students_df, use_container_width=True)
        else:
            for student in st.session_state.students_list:
                st.write(f"• {student}")
    else:
        st.info("No students enrolled yet.")

elif page == "📊 Records & Statistics":
    st.title("📊 Records & Statistics")
    st.markdown("---")
    
    # Show any pending messages
    show_messages()
    
    if st.session_state.attendance_df.empty:
        st.info("📝 No attendance records found. Start marking attendance to see statistics.")
    else:
        df = st.session_state.attendance_df.copy()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = len(df)
            st.metric("Total Records", total_records)
        
        with col2:
            unique_students = df['student_name'].nunique()
            st.metric("Unique Students", unique_students)
        
        with col3:
            today_count = len(df[df['date'] == date.today().strftime('%Y-%m-%d')])
            st.metric("Today's Attendance", today_count)
        
        with col4:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        st.markdown("---")
        
        # Date range filter
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime(df['date']).min().date(),
                max_value=date.today()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=date.today(),
                max_value=date.today()
            )
        
        # Filter data by date range
        mask = (pd.to_datetime(df['date']).dt.date >= start_date) & (pd.to_datetime(df['date']).dt.date <= end_date)
        filtered_df = df[mask]
        
        if not filtered_df.empty:
            # Attendance by date chart
            st.subheader("📈 Daily Attendance Trend")
            daily_counts = filtered_df.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            
            fig_line = px.line(
                daily_counts, 
                x='date', 
                y='count',
                title="Daily Attendance Count",
                markers=True
            )
            fig_line.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Attendances"
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Student attendance frequency
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👥 Student Attendance Frequency")
                student_counts = filtered_df['student_name'].value_counts().head(10)
                
                fig_bar = px.bar(
                    x=student_counts.index,
                    y=student_counts.values,
                    title="Top 10 Students by Attendance",
                    labels={'x': 'Student Name', 'y': 'Attendance Count'}
                )
                fig_bar.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.subheader("⏰ Attendance by Hour")
                filtered_df['hour'] = pd.to_datetime(filtered_df['time']).dt.hour
                hourly_counts = filtered_df['hour'].value_counts().sort_index()
                
                fig_hour = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Attendance Distribution by Hour",
                    labels={'x': 'Hour of Day', 'y': 'Attendance Count'}
                )
                st.plotly_chart(fig_hour, use_container_width=True)
            
            # Method usage
            st.subheader("📱 Detection Method Usage")
            method_counts = filtered_df['method'].value_counts()
            
            fig_pie = px.pie(
                values=method_counts.values,
                names=method_counts.index,
                title="Attendance Methods Used"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Raw data table
            st.subheader("📋 Attendance Records")
            
            # Search functionality
            search_student = st.text_input("🔍 Search by student name:")
            if search_student:
                display_df = filtered_df[filtered_df['student_name'].str.contains(search_student, case=False)]
            else:
                display_df = filtered_df
            
            # Sort options
            sort_column = st.selectbox(
                "Sort by:",
                ['timestamp', 'student_name', 'confidence'],
                index=0
            )
            sort_order = st.radio("Order:", ['Descending', 'Ascending'], horizontal=True)
            
            display_df = display_df.sort_values(
                sort_column, 
                ascending=(sort_order == 'Ascending')
            )
            
            st.dataframe(
                display_df[['student_name', 'date', 'time', 'confidence', 'method']],
                use_container_width=True
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Attendance Data",
                data=csv,
                file_name=f"attendance_records_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No records found for the selected date range.")

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Show any pending messages
    show_messages()
    
    # Model settings
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence score for face detection"
        )
    
    with col2:
        model_status = "✅ Loaded" if st.session_state.model else "❌ Not Loaded"
        st.metric("Model Status", model_status)
    
    # Model retraining
    st.subheader("🔄 Model Retraining")
    st.info("💡 Retrain the model when you have enrolled new students for better accuracy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input("Training Epochs", min_value=10, max_value=200, value=50)
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
    
    with col2:
        img_size = st.selectbox("Image Size", [416, 640, 832], index=1)
        use_gpu = st.checkbox("Use GPU (if available)", value=True)
    
    if st.button("🚀 Retrain Model", type="primary"):
        if len(st.session_state.students_list) > 0:
            with st.spinner("Retraining model... This may take a while."):
                try:
                    # Import the model trainer
                    from model_trainer import ModelTrainer
                    
                    trainer = ModelTrainer()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Preparing dataset...")
                    progress_bar.progress(20)
                    
                    # Prepare dataset
                    num_classes, num_images = trainer.prepare_yolo_dataset()
                    
                    status_text.text(f"Training on {num_classes} classes, {num_images} images...")
                    progress_bar.progress(40)
                    
                    # Train model
                    results = trainer.train_model(
                        epochs=epochs,
                        img_size=img_size,
                        batch_size=batch_size
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("Finalizing...")
                    
                    # Get best model path
                    best_model_path = trainer.get_best_model_path()
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    st.success("✅ Model retrained successfully!")
                    st.info(f"🔄 New model saved at: {best_model_path}")
                    st.info("💡 Restart the application to load the new model")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error retraining model: {str(e)}")
                    st.error("💡 Make sure you have enrolled students and required dependencies installed")
        else:
            st.warning("⚠️ No enrolled students found. Enroll students before retraining.")
    
    # Data management
    st.subheader("🗃️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Attendance Records", type="secondary"):
            if st.checkbox("I confirm I want to delete all attendance records"):
                try:
                    if os.path.exists(ATTENDANCE_CSV):
                        os.remove(ATTENDANCE_CSV)
                    st.session_state.attendance_df = load_attendance_data()
                    st.success("✅ Attendance records cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing records: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Enrolled Students", type="secondary"):
            if st.checkbox("I confirm I want to delete all enrolled students"):
                try:
                    if os.path.exists(STUDENTS_CSV):
                        os.remove(STUDENTS_CSV)
                    if os.path.exists(ENROLLED_FACES_DIR):
                        shutil.rmtree(ENROLLED_FACES_DIR)
                        os.makedirs(ENROLLED_FACES_DIR, exist_ok=True)
                    st.session_state.students_list = []
                    st.success("✅ Enrolled students cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing students: {str(e)}")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    info_data = {
        "Attendance Records File": ATTENDANCE_CSV,
        "Students List File": STUDENTS_CSV,
        "Enrolled Faces Directory": ENROLLED_FACES_DIR,
        "Model Path": MODEL_PATH,
        "Total Enrolled Students": len(st.session_state.students_list),
        "Total Attendance Records": len(st.session_state.attendance_df) if not st.session_state.attendance_df.empty else 0
    }
    
    for key, value in info_data.items():
        st.write(f"**{key}:** {value}")

st.markdown("---")
st.markdown("**📚 Student Attendance System** - Powered by YOLOv8 Face Detection")