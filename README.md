# Student Attendance System

A comprehensive web-based attendance monitoring system using YOLOv8 face detection technology.

## 🌟 Features

### 📝 Attendance Marking
- **Camera Capture**: Real-time face detection using webcam
- **Image Upload**: Upload images for batch attendance marking
- **Confidence Scoring**: Adjustable confidence thresholds for detection accuracy
- **Timestamp Logging**: Automatic date and time recording

### 👥 Student Enrollment
- **Multi-Photo Capture**: Capture multiple angles for better recognition
- **Batch Upload**: Upload multiple photos at once
- **Student Management**: Track student information (name, ID, class)
- **Organized Storage**: Automatic file organization by student

### 📊 Records & Analytics
- **Interactive Dashboard**: Visual attendance statistics
- **Date Range Filtering**: Analyze attendance patterns over time
- **Student Performance**: Individual attendance tracking
- **Data Export**: Download attendance records as CSV
- **Real-time Charts**: Daily trends, hourly patterns, method usage

### 🤖 AI Model Management
- **Model Retraining**: Retrain with newly enrolled students
- **Performance Monitoring**: Track detection confidence and accuracy
- **GPU Support**: Accelerated training when available
- **Model Versioning**: Keep track of training iterations

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
cd "Face Detection"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Access the Web Interface

Open your browser and navigate to: `http://localhost:8501`

## 📁 Project Structure

```
Face Detection/
├── app.py                      # Main Streamlit application
├── model_trainer.py            # Model retraining functionality
├── requirements.txt            # Python dependencies
├── analysis.ipynb             # Jupyter notebook for model development
├── attendance_records.csv     # Attendance data (auto-generated)
├── students_list.csv         # Student information (auto-generated)
├── enrolled_faces/           # Student photos directory
├── yolo_dataset/            # YOLO training data
└── runs/                    # Model training outputs
    └── detect/
        └── student_face_detection/
            └── weights/
                └── best.pt  # Trained model weights
```

## 🎯 Usage Guide

### Initial Setup

1. **Train the Model** (Optional - First Time):
   - Run the Jupyter notebook `analysis.ipynb` to train on your initial dataset
   - Or start with the pretrained YOLOv8n model for immediate use

2. **Enroll Students**:
   - Go to "👥 Enroll Student" page
   - Add student information
   - Capture/upload multiple photos per student
   - Photos are automatically organized and stored

3. **Mark Attendance**:
   - Go to "✅ Mark Attendance" page
   - Choose camera capture or image upload
   - System automatically detects and identifies faces
   - Confirm attendance with one click

4. **View Analytics**:
   - Go to "📊 Records & Statistics" page
   - View attendance trends and patterns
   - Export data for external analysis

### Model Retraining

When you enroll new students:

1. Go to "⚙️ Settings" page
2. Configure training parameters:
   - **Epochs**: 30-100 (more epochs = better accuracy, longer training)
   - **Batch Size**: 8-16 (higher = faster training, needs more memory)
   - **Image Size**: 640 recommended
3. Click "🚀 Retrain Model"
4. Wait for training completion (5-30 minutes depending on data size)
5. Restart the application to load the new model

## 📊 Data Management

### Attendance Records (`attendance_records.csv`)
- **student_name**: Identified student name
- **date**: Date of attendance
- **time**: Time of attendance
- **timestamp**: Full datetime
- **confidence**: Detection confidence score
- **method**: Camera or Upload

### Student Information (`students_list.csv`)
- **student_name**: Full name
- **student_id**: Optional ID number
- **class**: Class or grade
- **enrollment_date**: When student was enrolled
- **num_photos**: Number of photos captured

## 🛠️ Technical Details

### Face Detection Pipeline
1. **Image Preprocessing**: Resize and normalize input images
2. **YOLO Detection**: Use YOLOv8n for face localization
3. **Classification**: Identify specific students
4. **Confidence Filtering**: Filter low-confidence detections
5. **Result Visualization**: Draw bounding boxes and labels

### Model Architecture
- **Base Model**: YOLOv8n (Nano - optimized for speed)
- **Input Size**: 640x640 pixels
- **Classes**: Dynamic (based on enrolled students)
- **Training**: Transfer learning from COCO pretrained weights

## 🔧 Configuration

### Environment Variables
Create a `.env` file for custom configurations:

```env
MODEL_PATH=custom/path/to/model.pt
ATTENDANCE_CSV=custom_attendance.csv
ENROLLED_FACES_DIR=custom_faces_directory
```

### Performance Tuning
- **GPU Training**: Ensure CUDA is installed for faster training
- **Batch Size**: Increase for faster processing (limited by memory)
- **Confidence Threshold**: Adjust based on accuracy requirements
- **Image Quality**: Higher resolution improves accuracy but slower processing

## 🎨 Features Deep Dive

### Real-time Camera Capture
- Uses Streamlit's `camera_input` for seamless integration
- Automatic face detection on capture
- Instant attendance confirmation
- Multiple face detection in single image

### Batch Image Processing
- Support for JPG, PNG, BMP formats
- Drag-and-drop interface
- Preview before processing
- Batch attendance marking

### Interactive Analytics
- **Plotly Charts**: Interactive visualizations
- **Date Filtering**: Focus on specific time periods
- **Search Functionality**: Find specific students quickly
- **Export Options**: CSV download with custom date ranges

### Responsive Design
- **Mobile Friendly**: Works on tablets and phones
- **Column Layouts**: Optimized for different screen sizes
- **Progress Indicators**: Visual feedback for long operations

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. **Torch/PyTorch Warnings in Terminal**
```
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
```
**Solution**: These are harmless warnings. The app includes warning suppression, but you can also:
- Use the provided `start_app.bat` script
- Set environment variable: `set PYTHONWARNINGS=ignore` before running

#### 2. **Streamlit Deprecation Warnings**
```
The `use_column_width` parameter has been deprecated
```
**Solution**: All instances have been updated to `use_container_width`. Update your Streamlit version:
```bash
pip install --upgrade streamlit
```

#### 3. **Plotly Chart Errors**
```
AttributeError: 'Figure' object has no attribute 'update_xaxis'
```
**Solution**: Updated to use `fig.update_layout(xaxis_tickangle=45)` instead

#### 4. **Duplicate Attendance Records**
**Issue**: Students can be marked present multiple times per day
**Solution**: The app now prevents duplicate entries and shows appropriate messages:
- ✅ Success: First attendance of the day
- ⚠️ Warning: Already marked present today
- ❌ Error: System error occurred

#### 5. **Navigation Issues from Home Page**
**Problem**: Quick action buttons not working properly
**Solution**: Fixed navigation using session state instead of query parameters

#### 6. **Model Loading Failures**
**Common Causes**:
- Missing model file
- GPU memory issues
- Corrupted model weights

**Solutions**:
```bash
# Install/update ultralytics
pip install --upgrade ultralytics

# If GPU issues, force CPU usage
# The app automatically falls back to CPU if needed
```

#### 7. **Camera Not Working**
**Issues**:
- Browser permissions denied
- Multiple cameras detected
- WebRTC connection failed

**Solutions**:
- Enable camera permissions in browser
- Try refreshing the page
- Use image upload as alternative
- Check if other apps are using camera

#### 8. **File Permission Errors**
**Error**: `PermissionError: [Errno 13] Permission denied`
**Solutions**:
- Run as administrator (Windows)
- Check file/folder permissions
- Ensure no files are open in other programs

### Error Messages & Meanings

| Message | Meaning | Action |
|---------|---------|---------|
| ⚠️ Using pretrained model | No custom trained model found | Train model with your data |
| ❌ Model not loaded | Critical error in model loading | Check dependencies, restart app |
| ⚠️ {Student} already marked present today | Duplicate attendance attempt | No action needed, attendance logged |
| ✅ {Student} marked present! | Successful attendance logging | Continue normal operation |
| ⚠️ No faces detected | No faces found in image | Improve lighting, face visibility |

### Performance Optimization

#### For Better Speed:
1. **Reduce Image Size**: Use 640x640 instead of higher resolutions
2. **Lower Batch Size**: Use batch_size=8 for retraining if memory issues
3. **CPU vs GPU**: GPU faster for training, CPU sufficient for inference
4. **Close Other Apps**: Free up system memory

#### For Better Accuracy:
1. **Multiple Photos**: Capture 5-10 photos per student during enrollment
2. **Good Lighting**: Ensure well-lit faces during capture
3. **Different Angles**: Capture front, slightly left, slightly right poses
4. **Regular Retraining**: Retrain model when enrolling new students

### Installation Issues

#### Missing Dependencies:
```bash
# Full installation command
pip install streamlit ultralytics opencv-python pillow pandas numpy plotly matplotlib pyyaml pathlib2 torch torchvision
```

#### Version Conflicts:
```bash
# Create new environment
conda create -n attendance python=3.9
conda activate attendance
pip install -r requirements.txt
```

### Data Recovery

#### Lost Attendance Records:
- Check `attendance_records.csv` file
- Look for backup files with timestamps
- Data is auto-saved after each attendance

#### Lost Student Enrollments:
- Check `students_list.csv` and `enrolled_faces/` directory
- Re-enroll students if data corrupted
- Retrain model after re-enrollment

### Getting Help

If issues persist:
1. **Check Error Messages**: Read the specific error in terminal
2. **Update Dependencies**: Ensure latest versions installed
3. **Restart Application**: Close and restart Streamlit
4. **Clear Browser Cache**: Refresh browser completely
5. **Check File Paths**: Ensure all required files exist

## 🔮 Future Enhancements

- **Multi-class Detection**: Support for different types of faces (student, teacher, visitor)
- **Real-time Video**: Continuous video stream processing
- **Mobile App**: Native mobile application
- **Cloud Integration**: Cloud storage and processing
- **Advanced Analytics**: Machine learning insights on attendance patterns

## 📞 Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review the Jupyter notebook for model training details
3. Ensure all dependencies are properly installed

## 📄 License

This project is for educational and demonstration purposes. Customize and extend as needed for your specific requirements.

---

**Happy Teaching! 📚✨**