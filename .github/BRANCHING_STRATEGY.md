# Git Branching Strategy

## Branch Structure

This project follows **Git Flow** with feature branches for organized development.

### Main Branches

- **`main`**: Production-ready code only
  - Protected branch
  - Requires PR approval
  - All CI/CD checks must pass
  - Tagged releases (v1.0.0, v1.1.0, etc.)

- **`develop`**: Integration branch
  - Base for all feature branches
  - Contains latest development changes
  - Protected branch
  - Merge features here first

### Feature Branches

All feature branches are created from `develop` and merged back to `develop`.

#### Current/Active Branches

1. **`feature/ci-cd-pipeline`** ✅ (In Progress)
   - CI/CD infrastructure
   - Testing framework
   - Code quality tools
   - Status: Pushed, ready for PR

2. **`data-pipeline`** ⚠️ (Needs rename)
   - Should be: `feature/data-pipeline`
   - Data loading, preprocessing, splitting
   - Feature engineering
   - Status: Needs to merge CI/CD updates from develop

#### Planned Feature Branches

3. **`feature/class-imbalance`**
   - SMOTE implementation
   - Undersampling strategies
   - Class weight handling
   - Imbalanced data techniques

4. **`feature/model-training`**
   - Logistic Regression baseline
   - Random Forest implementation
   - XGBoost implementation
   - Model training pipelines

5. **`feature/model-evaluation`**
   - Metrics: Precision, Recall, F1, ROC-AUC
   - Confusion matrix visualization
   - Model comparison framework
   - Performance reporting

6. **`feature/mlflow-integration`**
   - MLflow setup
   - Experiment tracking
   - Model registry
   - Parameter logging

7. **`feature/api-development`**
   - FastAPI/Flask endpoints
   - Real-time prediction API
   - Request/response validation
   - API documentation (Swagger)

8. **`feature/monitoring`**
   - Model performance monitoring
   - Data drift detection
   - Alert system
   - Dashboard (Grafana/Streamlit)

9. **`feature/deployment`**
   - Docker containerization
   - Kubernetes manifests
   - CI/CD deployment pipeline
   - Production configuration

## Workflow

### Creating a New Feature

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name

# Work on feature
# ... make changes ...
git add .
git commit -m "feat: description"

# Push to remote
git push -u origin feature/your-feature-name

# Create PR to develop on GitHub
```

### Merging Strategy

```bash
# Before merging, ensure:
# 1. All CI/CD checks pass ✅
# 2. Code review approved ✅
# 3. No merge conflicts ✅
# 4. Tests pass locally ✅

# Merge via GitHub PR or:
git checkout develop
git merge --no-ff feature/your-feature-name
git push origin develop
```

### Release Strategy

```bash
# When ready for production release
git checkout main
git merge --no-ff develop
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin main --tags
```

## Branch Naming Conventions

- **Feature**: `feature/descriptive-name`
- **Bugfix**: `bugfix/issue-description`
- **Hotfix**: `hotfix/critical-fix`
- **Release**: `release/v1.0.0`
- **Experimental**: `experiment/idea-name`

## Current Priority

### Phase 1: Foundation ✅ (Completed)
- [x] Data pipeline
- [x] CI/CD infrastructure

### Phase 2: ML Development (Next)
1. `feature/class-imbalance` - Handle imbalanced data
2. `feature/model-training` - Build baseline models
3. `feature/model-evaluation` - Evaluation framework
4. `feature/mlflow-integration` - Experiment tracking

### Phase 3: Production (Later)
5. `feature/api-development` - REST API
6. `feature/monitoring` - Monitoring system
7. `feature/deployment` - Production deployment

## Branch Protection Rules

### For `main`:
- Require pull request reviews (2 approvers)
- Require status checks to pass
- Require branches to be up to date
- No direct pushes
- No force pushes

### For `develop`:
- Require pull request reviews (1 approver)
- Require status checks to pass
- No force pushes
- Delete head branches after merge

## Action Items

### Immediate
- [ ] Rename `data-pipeline` to `feature/data-pipeline`
- [ ] Merge `feature/ci-cd-pipeline` → `develop`
- [ ] Merge `feature/data-pipeline` → `develop`

### Next Sprint
- [ ] Start `feature/class-imbalance`
- [ ] Start `feature/model-training`

## Notes

- **Always** create feature branches from `develop`
- **Never** merge directly to `main` (use PR)
- **Delete** feature branches after merging
- **Tag** all production releases
- **Rebase** before merging if commits are messy
