# Deployment Guide

This guide provides recommendations for deploying applications that use the FinanceToolkit in various environments, including container orchestration platforms like Kubernetes.

## Table of Contents

1. [Overview](#overview)
2. [Container Deployment](#container-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Best Practices](#best-practices)
5. [Example Configurations](#example-configurations)

## Overview

The FinanceToolkit is a Python library designed for financial analysis and calculations. It is not a standalone web service but rather a package that you integrate into your Python applications. This guide covers how to deploy applications that use the FinanceToolkit in containerized and orchestrated environments.

## Container Deployment

### Docker

To containerize an application using the FinanceToolkit, create a Dockerfile:

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables (use secrets management in production)
ENV FINANCIAL_MODELING_PREP_KEY=""

# Run your application
CMD ["python", "your_app.py"]
```

**requirements.txt example:**
```
financetoolkit>=2.0.6
pandas>=2.2
scikit-learn>=1.6
requests>=2.32
```

Build and run:
```bash
docker build -t financetoolkit-app .
docker run -e FINANCIAL_MODELING_PREP_KEY=your_api_key financetoolkit-app
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured to access your cluster
- Container registry access (Docker Hub, GCR, ECR, etc.)

### Basic Deployment

#### 1. Store API Keys as Secrets

```bash
kubectl create secret generic financetoolkit-secrets \
  --from-literal=api-key=YOUR_FINANCIAL_MODELING_PREP_KEY
```

#### 2. Create a Deployment

Save as `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financetoolkit-app
  labels:
    app: financetoolkit
spec:
  replicas: 3
  selector:
    matchLabels:
      app: financetoolkit
  template:
    metadata:
      labels:
        app: financetoolkit
    spec:
      containers:
      - name: financetoolkit-app
        image: your-registry/financetoolkit-app:latest
        env:
        - name: FINANCIAL_MODELING_PREP_KEY
          valueFrom:
            secretKeyRef:
              name: financetoolkit-secrets
              key: api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        # Uncomment if your app exposes an HTTP endpoint
        # ports:
        # - containerPort: 8080
        #   name: http
```

Apply the deployment:
```bash
kubectl apply -f deployment.yaml
```

### Job-Based Deployment (for scheduled tasks)

If you're running periodic financial analysis tasks, use a CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: financetoolkit-analysis
spec:
  schedule: "0 2 * * *"  # Run daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: financetoolkit-job
            image: your-registry/financetoolkit-app:latest
            env:
            - name: FINANCIAL_MODELING_PREP_KEY
              valueFrom:
                secretKeyRef:
                  name: financetoolkit-secrets
                  key: api-key
            resources:
              requests:
                memory: "512Mi"
                cpu: "500m"
              limits:
                memory: "1Gi"
                cpu: "1000m"
          restartPolicy: OnFailure
```

### Service Configuration (if exposing an API)

If your application exposes a REST API or web interface:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: financetoolkit-service
spec:
  selector:
    app: financetoolkit
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: LoadBalancer
```

## Best Practices

### 1. Resource Management

The FinanceToolkit can be resource-intensive when processing large datasets:

- **Memory**: Allocate at least 512Mi for basic operations, 1-2Gi for heavy analysis
- **CPU**: 250m-500m for basic operations, 1-2 cores for intensive calculations
- **Use Horizontal Pod Autoscaling (HPA)** for variable workloads:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: financetoolkit-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: financetoolkit-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 2. API Rate Limiting

The FinanceToolkit relies on external APIs (FinancialModelingPrep, Yahoo Finance):

- Free tier: 250 requests/day
- Implement caching to reduce API calls
- Use `use_cached_data=True` in the Toolkit initialization
- Consider using persistent volumes for cache storage:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: financetoolkit-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

Mount in your deployment:
```yaml
        volumeMounts:
        - name: cache-storage
          mountPath: /app/cache
      volumes:
      - name: cache-storage
        persistentVolumeClaim:
          claimName: financetoolkit-cache
```

### 3. Security Best Practices

- **Never hardcode API keys** - use Kubernetes Secrets
- Use **RBAC** to restrict access to secrets
- Consider using **external secrets management** (HashiCorp Vault, AWS Secrets Manager, etc.)
- **Network policies** to restrict egress to only necessary external APIs:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: financetoolkit-netpol
spec:
  podSelector:
    matchLabels:
      app: financetoolkit
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 53  # DNS
  - to:
    - podSelector: {}
  - ports:
    - protocol: TCP
      port: 443  # HTTPS for API calls
```

### 4. Monitoring and Logging

Implement proper monitoring:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: financetoolkit-config
data:
  logging.conf: |
    [loggers]
    keys=root

    [handlers]
    keys=consoleHandler

    [formatters]
    keys=simpleFormatter

    [logger_root]
    level=INFO
    handlers=consoleHandler

    [handler_consoleHandler]
    class=StreamHandler
    level=INFO
    formatter=simpleFormatter
    args=(sys.stdout,)

    [formatter_simpleFormatter]
    format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### 5. Health Checks

Implement health checks if running a web service:

```yaml
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Example Configurations

### Example 1: Scheduled Financial Analysis Job

Complete example for running daily financial analysis:

```yaml
---
apiVersion: v1
kind: Secret
metadata:
  name: financetoolkit-secrets
type: Opaque
stringData:
  api-key: YOUR_API_KEY_HERE

---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-stock-analysis
spec:
  schedule: "0 1 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: analysis
            image: your-registry/financetoolkit-app:latest
            env:
            - name: FINANCIAL_MODELING_PREP_KEY
              valueFrom:
                secretKeyRef:
                  name: financetoolkit-secrets
                  key: api-key
            - name: TICKERS
              value: "AAPL,MSFT,GOOGL,AMZN"
            volumeMounts:
            - name: cache
              mountPath: /app/cache
            resources:
              requests:
                memory: "1Gi"
                cpu: "500m"
              limits:
                memory: "2Gi"
                cpu: "1000m"
          volumes:
          - name: cache
            persistentVolumeClaim:
              claimName: financetoolkit-cache
          restartPolicy: OnFailure

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: financetoolkit-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### Example 2: REST API Service

If you've built a REST API using the FinanceToolkit (e.g., with FastAPI or Flask):

```yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financetoolkit-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: financetoolkit-api
  template:
    metadata:
      labels:
        app: financetoolkit-api
    spec:
      containers:
      - name: api
        image: your-registry/financetoolkit-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: FINANCIAL_MODELING_PREP_KEY
          valueFrom:
            secretKeyRef:
              name: financetoolkit-secrets
              key: api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
apiVersion: v1
kind: Service
metadata:
  name: financetoolkit-api
spec:
  selector:
    app: financetoolkit-api
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: financetoolkit-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: financetoolkit-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Example Python Application Code

Here's a sample application structure that works well in Kubernetes:

**app.py:**
```python
import os
from financetoolkit import Toolkit

def main():
    # Get configuration from environment variables
    api_key = os.environ.get('FINANCIAL_MODELING_PREP_KEY')
    tickers = os.environ.get('TICKERS', 'AAPL,MSFT').split(',')
    cache_dir = os.environ.get('CACHE_DIR', '/app/cache')
    
    # Initialize toolkit with caching enabled
    companies = Toolkit(
        tickers=tickers,
        api_key=api_key,
        use_cached_data=cache_dir,
        start_date="2020-01-01"
    )
    
    # Perform analysis
    ratios = companies.ratios.collect_profitability_ratios()
    print(ratios)
    
    # Save results
    ratios.to_csv(f'{cache_dir}/results.csv')

if __name__ == "__main__":
    main()
```

## Additional Resources

- [FinanceToolkit Documentation](https://www.jeroenbouma.com/projects/financetoolkit)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## Contributing

If you have improvements or additional deployment scenarios, please contribute to the [FinanceToolkit repository](https://github.com/JerBouma/FinanceToolkit).

## Contact

For questions about deployment or the FinanceToolkit:
- **Website**: https://jeroenbouma.com/
- **LinkedIn**: https://www.linkedin.com/in/boumajeroen/
- **Email**: jer.bouma@gmail.com
