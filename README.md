# Face Identification System:FastAPI & Streamlit

A real-time **face identification application** built with **FastAPI** and
**Streamlit**, using `face_recognition` and `dlib` for face detection and
128-dimensional face embedding generation.

The application allows a user to upload an image, detect a face, compare its
facial embedding against a set of reference images, and return the closest
matching identity when the similarity threshold is satisfied.

---

## Overview

The project separates the application into a backend API and an interactive
frontend:

```text
                         USER
                          │
                          ▼
                Streamlit Web Interface
                     Port 8501
                          │
                          │ HTTP POST
                          ▼
                   FastAPI Backend
                     Port 8000
                          │
                          ▼
                face_recognition
                          │
                          ▼
                    dlib Models
                          │
                          ▼
              128-Dimensional Embedding
                          │
                          ▼
                 Euclidean Distance
                          │
                  ┌───────┴───────┐
                  │               │
              Match Found      No Match
                  │               │
                  ▼               ▼
              Identity          Unknown
