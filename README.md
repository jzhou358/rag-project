# COSC 6376 Cloud Computing - Fall 2025 -  Final Project
## **Deployment and Optimization of a RAG-Enhanced LLM Agent via DevOps Pipeline**  
### **Author: Junchao Zhou – 2401060**

---

## **Project Overview**

This project implements a **Retrieval-Augmented Generation (RAG) Large Language Model Agent** and deploys it on **AWS ECS Fargate** using a fully automated **CI/CD DevOps pipeline** (GitHub → CodePipeline → CodeBuild → ECS).

The application allows users to **query a PDF document** using:

- Chunk-level document embeddings  
- Chapter-summary embeddings  
- Book-quotes embeddings  
- An RAG pipeline  

The entire system is deployed as a **scalable containerized service**, accessible through a public URL via an **Application Load Balancer (ALB)**.

---

## **Live Application URL**

**http://rag-project-alb-977672821.us-east-1.elb.amazonaws.com/**