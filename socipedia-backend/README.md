# Socipedia Django Backend

A full-featured social media backend built with Django REST Framework, featuring **AI-powered content moderation**, automatic image compression, JWT authentication, real-time WebSockets, and comprehensive social media functionality.

## 🚀 Features

- **AI Content Moderation** 🤖:
  - 🔞 Nude image detection using K-means clustering (98% accuracy)
  - 🚫 Hate speech detection using OCR and keyword matching
  - 💬 Text toxicity analysis using sentiment analysis
  - ⚡ GPU acceleration support (automatic fallback to CPU)
- **Authentication**: JWT-based authentication with registration and login
- **User Management**: Custom user model with profile pictures and friend management
- **Posts**: Create, read, update, delete posts with image support
- **Comments**: Full commenting system with like functionality
- **Real-time Notifications**: WebSocket support for live updates
- **Image Compression**: Automatic image compression for optimal storage and performance
- **Like System**: Like/unlike posts and comments
- **Friend System**: Add/remove friends with real-time notifications
- **Image Serving**: Dedicated API for serving images to frontend
- **CORS Support**: Frontend integration ready

## 🔧 Image Compression & Content Moderation

### Image Compression

All uploaded images are automatically compressed:

- **Profile Pictures**: Max 800x800px, 85% quality
- **Post Images**: Max 1200x1200px, 80% quality
- **Average Compression**: 70-95% size reduction
- **Format Support**: JPEG, PNG with smart format handling

### AI Content Moderation

**Nude Image Detection**:

- K-means clustering (5 clusters) with MobileNetV2 features
- Training data: 18,667 images (98.1% accuracy for adult cluster)
- Automatic blocking of inappropriate images

**Hate Speech Detection**:

- EasyOCR text extraction from images
- 60+ hate/violence/vulgar keywords database
- Automatic blocking of images with offensive text

**Text Toxicity Detection**:

- Combined sentiment + toxicity analysis
- Thresholds: Negative > 0.8 AND Toxicity > 0.7
- Blocks toxic/offensive text content

**GPU Acceleration**:

- Automatic GPU detection (NVIDIA CUDA)
- Fallback to CPU if no GPU available
- EasyOCR GPU acceleration: 2-3x faster processing

## 📋 API Endpoints

### Authentication

- `POST /api/register/` - User registration
- `POST /api/login/` - User login
- `POST /api/token/refresh/` - Refresh JWT token

### Users

- `GET /api/users/` - List users
- `GET /api/users/{id}/` - Get user details
- `PATCH /api/users/{id}/` - Update user
- `PATCH /api/users/{id}/upload_picture/` - Upload profile picture
- `POST /api/users/{id}/add_friend/` - Add friend
- `POST /api/users/{id}/remove_friend/` - Remove friend

### Posts

- `GET /api/posts/` - List posts
- `POST /api/posts/` - Create post (with image)
- `GET /api/posts/{id}/` - Get post details
- `PATCH /api/posts/{id}/` - Update post
- `PATCH /api/posts/{id}/update_post/` - Custom update (description + image)
- `DELETE /api/posts/{id}/` - Delete post
- `POST /api/posts/{id}/like/` - Like/unlike post

### Comments

- `GET /api/comments/` - List comments
- `POST /api/comments/` - Create comment
- `GET /api/comments/{id}/` - Get comment
- `PATCH /api/comments/{id}/` - Update comment
- `DELETE /api/comments/{id}/` - Delete comment
- `POST /api/comments/{id}/like/` - Like/unlike comment

### Images

- `GET /api/images/{filename}` - Serve images to frontend

### Notifications

- `GET /api/notifications/` - Get user notifications
- `WebSocket /ws/notifications/{user_id}/` - Real-time notification updates

## 📋 Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Redis server (for WebSocket/channels)
- Optional: NVIDIA GPU with CUDA support (for faster AI moderation)

## 🛠️ Local Development

### Quick Setup (Recommended)

#### Windows:

```bash
setup.bat
```

#### Linux/Mac:

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:

- Detect if you have an NVIDIA GPU
- Install appropriate dependencies (GPU or CPU versions)
- Verify the installation

### Manual Setup

#### Option 1: CPU-Only Systems (No NVIDIA GPU)

1. **Create virtual environment**:

   ```bash
   python -m venv .venv
   ```

2. **Activate virtual environment**:

   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements-cpu.txt
   ```

4. **Run migrations**:

   ```bash
   python manage.py migrate
   ```

5. **Start Redis** (in separate terminal):

   ```bash
   redis-server
   ```

6. **Start development server**:
   ```bash
   python manage.py runserver
   ```

#### Option 2: GPU Systems (NVIDIA GPU Available)

1. **Create virtual environment**:

   ```bash
   python -m venv .venv
   ```

2. **Activate virtual environment**:

   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install base requirements**:

   ```bash
   pip install -r requirements-cpu.txt
   ```

4. **Uninstall CPU-only PyTorch**:

   ```bash
   pip uninstall torch torchvision torchaudio -y
   ```

5. **Install GPU-enabled PyTorch** (CUDA 11.8):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   ⚠️ Note: This downloads ~2.8GB

6. **Verify GPU is detected**:

   ```bash
   # Linux/Mac
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   
   # Windows (PowerShell) - use backticks to escape quotes
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else `"None`"}')"
   
   # Alternative: Create check_gpu.py with the code and run python check_gpu.py
   ```

7. **Run migrations**:

   ```bash
   python manage.py migrate
   ```

8. **Start Redis** (in separate terminal):

   ```bash
   redis-server
   ```

9. **Start development server**:
   ```bash
   python manage.py runserver
   ```

The API will be available at `http://localhost:8000/api/`

### Testing Image Moderation

**Test Nude Image Detection**:

```bash
curl -X POST http://127.0.0.1:8000/api/posts/ \
  -H "Content-Type: multipart/form-data" \
  -F "description=Test post" \
  -F "picturePath=@/path/to/test/image.jpg"
```

**Expected Response** (for inappropriate image):

```json
{
  "picturePath": [
    "Image contains inappropriate content and cannot be uploaded."
  ]
}
```

**Expected Response** (for appropriate image):

```json
{
  "id": 123,
  "description": "Test post",
  "picturePath": "username_image.jpg",
  ...
}
```

## 🚀 Deployment (Render)

### Environment Variables

Set these in your Render dashboard:

- `SECRET_KEY` - Django secret key
- `DEBUG` - Set to "False"
- `DATABASE_URL` - PostgreSQL connection string (auto-provided by Render)

### Build Command

```bash
./build.sh
```

### Start Command

```bash
gunicorn socipedia.wsgi:application
```

## 📚 Documentation

- **[GPU Installation Guide](GPU_INSTALLATION_GUIDE.md)** - Detailed GPU setup instructions
- **[GPU Setup Summary](GPU_SETUP_SUMMARY.md)** - Current GPU configuration
- `api-localhost-endpoints.json` - Complete API documentation with curl examples
- `api-endpoints-with-images.json` - Production API endpoints
- `IMAGE_COMPRESSION_IMPLEMENTATION.md` - Detailed image compression guide
- `PATCH_AND_IMAGES_IMPLEMENTATION.md` - PATCH API and image serving guide
- `RENDER_DEPLOYMENT_GUIDE.md` - Production deployment guide

## ⚡ Performance

### With NVIDIA GPU:

- **First image moderation**: ~10-15 seconds (model loading)
- **Subsequent images**: ~3-5 seconds
- **EasyOCR**: GPU accelerated ⚡ (2-3x faster)
- **TensorFlow**: CPU mode (still fast)

### CPU-Only:

- **First image moderation**: ~15-20 seconds (model loading)
- **Subsequent images**: ~5-10 seconds
- **EasyOCR**: CPU mode (functional but slower)
- **TensorFlow**: CPU mode

## 🐛 Troubleshooting

### "No GPU found, using CPU"

- **On GPU system**: Follow GPU setup in [GPU Installation Guide](GPU_INSTALLATION_GUIDE.md)
- **On CPU system**: This is normal - everything works, just slower

### "ModuleNotFoundError: No module named 'joblib'"

```bash
pip install -r requirements-cpu.txt
```

### "Redis connection error"

- Start Redis server: `redis-server`
- Check Redis is running: `redis-cli ping` (should return `PONG`)
- Or disable channels in `settings.py` if not using WebSockets

### WebSocket connection fails

- Verify Redis is running
- Check WebSocket URL: `ws://127.0.0.1:8000/ws/notifications/{user_id}/`
- Ensure JWT token is valid

### Slow image processing

- **First time**: Models need to load from disk (~10-20 seconds)
- **Subsequent**: Should be faster (~3-10 seconds), models cached in memory
- **Still slow**: Check if GPU is detected (see GPU setup above)

### Image moderation not working

- Check models exist in `ml_models/` directory:
  - `kmeans_model.joblib`
  - `dataset_features.npy`
  - `valid_image_paths.txt`
- Check server logs for error messages
- Verify dependencies installed: `pip list | grep -E "(tensorflow|easyocr|joblib)"`

## 🗂️ Project Structure

```
backend/
├── api/                          # Main API app
│   ├── models.py                 # Database models
│   ├── views.py                  # API endpoints
│   ├── serializers.py            # DRF serializers with validation
│   ├── image_moderation.py       # ⭐ AI moderation logic
│   ├── consumers.py              # WebSocket consumers
│   └── urls.py                   # API routes
├── socipedia/                    # Project settings
│   ├── settings.py               # Django settings
│   ├── urls.py                   # Root URL config
│   └── asgi.py                   # ASGI config for WebSockets
├── ml_models/                    # ⭐ Trained ML models
│   ├── kmeans_model.joblib       # K-means clustering model
│   ├── dataset_features.npy      # Feature vectors (18,667 images)
│   └── valid_image_paths.txt     # Training labels
├── combined_model/               # ⭐ Text moderation
│   └── combined_model/
│       └── infer.py              # Toxicity detection
├── requirements-cpu.txt          # ⭐ CPU-only dependencies
├── req_with_gpu.txt              # Same as CPU (for compatibility)
├── setup.bat                     # ⭐ Windows setup script
├── setup.sh                      # ⭐ Linux/Mac setup script
├── GPU_INSTALLATION_GUIDE.md     # ⭐ GPU setup guide
├── GPU_SETUP_SUMMARY.md          # ⭐ GPU configuration
├── manage.py                     # Django management script
└── db.sqlite3                    # SQLite database
```

## 🔒 Security & Performance

- JWT authentication with refresh tokens
- User ownership validation for all operations
- **AI-powered content moderation** (nudity, hate speech, toxicity)
- Automatic image compression (70-95% size reduction)
- GPU acceleration for faster processing (when available)
- Real-time WebSocket notifications with Redis
- CORS properly configured
- Static file serving via Whitenoise

## 🎯 Quick Start Summary

```bash
# 1. Clone and navigate
cd backend

# 2. Run automated setup
setup.bat          # Windows
# or
./setup.sh         # Linux/Mac

# 3. Apply database migrations
python manage.py migrate

# 4. Start Redis (in separate terminal)
redis-server

# 5. Start Django server
python manage.py runserver

# 6. Visit http://127.0.0.1:8000/api/
```

## 🤝 For Your Friends (CPU-Only Laptops)

Your friends **without NVIDIA GPUs** can use this backend! Just follow these steps:

1. **Install dependencies**:

   ```bash
   pip install -r requirements-cpu.txt
   ```

2. **Run migrations**:

   ```bash
   python manage.py migrate
   ```

3. **Start Redis**:

   ```bash
   redis-server
   ```

4. **Start server**:
   ```bash
   python manage.py runserver
   ```

✅ **Everything works on CPU!** The only difference is slightly slower image moderation (5-10 seconds instead of 3-5 seconds).

## 📝 Notes

- **GPU vs CPU**: The code automatically detects available hardware and adjusts accordingly
- **PyTorch GPU**: Cannot be installed via `requirements.txt` - use setup scripts or manual GPU setup
- **Redis Required**: For WebSocket notifications (can be disabled in settings if not needed)
- **First Run**: Slower due to model loading - subsequent requests are cached

---

**Made with ❤️ by Prathamesh Patil**
