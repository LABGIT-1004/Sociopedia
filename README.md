# Sociopedia
Sociopedia combines social networking with AI sentiment analysis. It allows users to share posts and comments while analyzing emotions behind the content, promoting positive interactions and reducing negativity for a healthier digital community


# Sociopedia

A full-stack social media application built with Django (backend) and React/Vite (frontend), featuring user authentication, posts, comments, messaging, and GPU-accelerated features.

## Features

- User registration and authentication (Google OAuth, Auth0)
- Post creation with image uploads and compression
- Comments and likes on posts
- Real-time messaging with WebSockets
- Friend requests and notifications
- Image moderation using GPU acceleration
- Responsive UI with Tailwind CSS

## Prerequisites

- Python 3.8+
- Node.js 16+
- PostgreSQL or SQLite (default)
- GPU support (optional, for image moderation)

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd sociopedia-backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. For GPU support (optional):
   - Follow the instructions in `GPU_SETUP_SUMMARY.md`
   - Install GPU-specific requirements: `pip install -r req_with_gpu.txt`

5. Run database migrations:
   ```bash
   python manage.py migrate
   ```

6. Create a superuser (optional):
   ```bash
   python manage.py createsuperuser
   ```

7. Start the Django development server:
   ```bash
   python manage.py runserver
   ```

The backend will be running on `http://localhost:8000`.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd sociopedia-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be running on `http://localhost:5173` (default Vite port).

## Configuration

### Environment Variables

Create `.env` files in both `sociopedia-backend` and `sociopedia-frontend` directories.

#### Backend (.env)
```
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3  # or PostgreSQL URL
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
AUTH0_DOMAIN=your-auth0-domain
AUTH0_CLIENT_ID=your-auth0-client-id
AUTH0_CLIENT_SECRET=your-auth0-client-secret
```

#### Frontend (.env)
```
VITE_API_URL=http://localhost:8000/api
VITE_GOOGLE_CLIENT_ID=your-google-client-id
VITE_AUTH0_DOMAIN=your-auth0-domain
VITE_AUTH0_CLIENT_ID=your-auth0-client-id
```

## Running the Project

1. Start the backend server (as described above)
2. Start the frontend server (as described above)
3. Open your browser and navigate to `http://localhost:5173`

## API Endpoints

The backend provides REST API endpoints for:
- User management
- Posts and comments
- Messaging
- Friend requests
- Notifications

See `api-endpoints-with-images.json` for detailed endpoint documentation.

## Development

### Backend
- Run tests: `python manage.py test`
- Code formatting: Use Black and isort
- Linting: Use flake8

### Frontend
- Run tests: `npm test`
- Code formatting: `npm run format`
- Linting: `npm run lint`

## Deployment

### Backend
- Use `gunicorn` for production
- Configure static files serving
- Set up PostgreSQL database

### Frontend
- Build for production: `npm run build`
- Serve static files from `dist/` directory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
