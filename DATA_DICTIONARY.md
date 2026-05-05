# Data Dictionary
## Matrix Mining in Academic Ecosystems Dataset

**Generated:** 2026-05-04 23:22:47
**Source:** Synthetic data based on ISU academic patterns

### Dataset Specifications
- **Rows (Students):** 15
- **Columns (Courses):** 10000
- **Total Ratings:** 7526
- **Sparsity:** 94.98%

### Files

#### 1. students.csv
User demographic and academic information
- `user_id`: Unique user identifier (STU_001 to STU_015)
- `major`: User's major program
- `gpa`: Grade Point Average (2.5-4.0)
- `year`: Academic year (1-5)

#### 2. courses.csv
Item catalog information
- `item_id`: Unique item identifier (DEPT###_#####)
- `department`: Academic department code
- `level`: Item level (100-500)
- `difficulty`: Perceived difficulty (1-5)
- `workload`: Expected workload hours (1-5)

#### 3. ratings.csv
User item ratings (sparse format)
- `user_id`: User identifier
- `item_id`: Item identifier
- `rating`: Rating score (1-5 scale)

#### 4. ratings_matrix.npy
Dense matrix format (students × courses)
- Rows: Students
- Columns: Courses
- Values: Ratings (0 = not rated)

### Data Collection Methodology
This synthetic dataset simulates realistic academic item rating patterns using:
1. Latent factor models to create user-item affinity patterns
2. Department and level-based item clustering
3. Realistic sparsity (students only rate courses they've taken)
4. Gaussian noise to simulate rating variability

### Usage Notes
- Zero values in the matrix indicate unrated courses (not low ratings)
- Ratings follow a 1-5 scale (1=Poor, 5=Excellent)
- Dataset designed for collaborative filtering and matrix factorization experiments
