# Neural Network for GWAS: One-time training with 5-fold cross-validation
# performs 5-fold cross-validation, and computes feature importance scores for GWAS.

# Load required libraries
library(keras)        # For building and training the neural network
library(tensorflow)   # Backend for Keras
library(data.table)   # For efficient data handling
library(ggplot2)      # For plotting training history
library(dplyr)        # For data manipulation
library(caret)        # For creating cross-validation folds

# Step 1: Load and preprocess data
# Load genotype (SNP) data
myGD <- read.table(file = "http://zzlab.net/GAPIT/data/mdp_numeric.txt", head = TRUE)
# Load SNP information
myGM <- read.table(file = "http://zzlab.net/GAPIT/data/mdp_SNP_information.txt", head = TRUE)
# Load covariate data (environmental factors)
myCV <- read.table(file = "http://zzlab.net/GAPIT/data/mdp_env.txt", head = TRUE)
# Load phenotype data (simulated)
myY <- read.csv(file = "http://zzlab.net/StaGen/2025/Data/my_y.csv", head = TRUE)
# Load QTN data (from simulation)
myQTN <- read.table(file = "http://zzlab.net/StaGen/2025/Data/my_QTN.txt", head = TRUE)

# Prepare genotype data
genotypes <- myGD
genotypes <- genotypes[, -1]  # Remove ID column

# Perform PCA on genotype data to capture population structure
pcs <- prcomp(myGD[,-1])  # Exclude ID column
# Extract first three PCA components
pca <- as.data.frame(pcs$x[, 1:3])

# Prepare covariate data
covariates <- myCV
covariates <- covariates[, -1]  # Remove ID column

# Combine PCA components and environmental covariates
covariates <- cbind(pca, covariates)

# Combine genotypes and covariates into input matrix X
X <- as.matrix(cbind(genotypes, covariates))

# Normalize the input data (neural networks perform better with scaled inputs)
X <- scale(X)

# Check for missing values in scaled data
sum(is.na(X))  
# Replace any NAs with 0 (alternative: use imputation methods)
X[is.na(X)] <- 0

# Load phenotype data 
phenotypes <- myY
y <- phenotypes$Sim                   

# Normalize the phenotype data
y <- scale(y)

# Step 2: Set up 5-fold cross-validation
# Create indices for 5 folds, ensuring balanced splits
set.seed(123)  # For reproducibility
folds <- createFolds(y, k = 5, returnTrain = TRUE)

# Initialize vector to store importance scores across folds
importance_scores <- numeric(ncol(X))

# Step 3: Train the neural network with cross-validation
for (fold_idx in seq_along(folds)) {
  cat("Training fold", fold_idx, "of 5\n")
  
  # Split data into training and validation sets
  train_idx <- folds[[fold_idx]]                  # Training indices
  val_idx <- setdiff(1:nrow(X), train_idx)        # Validation indices
  
  X_train <- X[train_idx, ]                       # Training features
  y_train <- y[train_idx]                         # Training labels
  X_val <- X[val_idx, ]                           # Validation features
  y_val <- y[val_idx]                             # Validation labels
  
  # Step 4: Build the neural network model
  model <- keras_model_sequential() %>%
    # First dense layer: 48 units, ReLU activation, L2 regularization
    layer_dense(units = 48, activation = "relu", input_shape = ncol(X), 
                kernel_regularizer = regularizer_l2(0.08)) %>%
    # Batch normalization to stabilize training
    layer_batch_normalization() %>%
    # Dropout (25%) to prevent overfitting
    layer_dropout(rate = 0.25) %>%
    
    # Second dense layer: 24 units, ReLU activation, L2 regularization
    layer_dense(units = 24, activation = "relu", 
                kernel_regularizer = regularizer_l2(0.08)) %>%
    # Second batch normalization
    layer_batch_normalization() %>%
    # Second dropout (25%)
    layer_dropout(rate = 0.25) %>%
    
    # Output layer: single unit for phenotype prediction
    layer_dense(units = 1)
  
  # Step 5: Compile the model
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.0003),  # Adam optimizer with low learning rate
    loss = "mean_squared_error"                          # MSE loss for regression
  )
  
  # Step 6: Define callbacks
  # Learning rate scheduler: reduce learning rate if validation loss plateaus
  lr_scheduler <- callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.3,       # Reduce learning rate by 30%
    patience = 5,       # Wait 5 epochs before reducing
    min_lr = 1e-6       # Minimum learning rate
  )
  
  # Early stopping: stop training if validation loss stops improving
  early_stopping <- callback_early_stopping(
    monitor = "val_loss",
    patience = 5,       # Wait 5 epochs before stopping
    restore_best_weights = TRUE  # Restore weights from best epoch
  )
  
  # Step 7: Train the model
  history <- model %>% fit(
    X_train, y_train,
    epochs = 500,          # Maximum number of epochs
    batch_size = 16,       # Batch size
    validation_data = list(X_val, y_val),  # Validation data
    callbacks = list(early_stopping, lr_scheduler),  # Apply callbacks
    verbose = 1            # Show training progress
  )
  
  # Step 8: Plot training history
  # Visualizes training and validation loss over epochs
  plot(history)
  
  # Step 9: Compute feature importance scores using gradients
  compute_gradients <- function(model, X) {
    # Convert input to TensorFlow tensor
    X_tensor <- tf$convert_to_tensor(X, dtype = tf$float32)
    # Use GradientTape to compute gradients of predictions w.r.t. inputs
    with(tf$GradientTape() %as% tape, {
      tape$watch(X_tensor)
      predictions <- model(X_tensor, training = FALSE)
    })
    # Calculate gradients
    gradients <- tape$gradient(predictions, X_tensor)
    gradients <- as.matrix(gradients)
    # Replace NA or infinite values with 0
    gradients[is.na(gradients) | is.infinite(gradients)] <- 0
    # Compute mean absolute gradient for each feature
    importance_scores <- colMeans(abs(gradients))
    
    # Normalize importance scores to [0, 1]
    importance_scores <- (importance_scores - min(importance_scores)) / 
      (max(importance_scores) - min(importance_scores))
    return(importance_scores)
  }
  
  # Compute importance scores for this fold
  fold_importance_scores <- compute_gradients(model, X)
  # Accumulate scores across folds
  importance_scores <- importance_scores + fold_importance_scores
}

# Step 10: Average importance scores across folds
importance_scores <- importance_scores / length(folds)
names(importance_scores) <- colnames(X)

# Step 11: Summarize and save results
# Print summary of importance scores
cat("Summary of importance scores:\n")
print(summary(importance_scores))

# Save importance scores to a CSV file
write.csv(data.frame(Feature = names(importance_scores), Importance_Score = importance_scores),
          file = "importance_scores_gwas.csv",
          row.names = FALSE)

# Step 12: Visulization
# Convert importance score to P value like
imp<-importance_scores
myP=1/(exp(20*imp))

# Manhattan plot
par(mfrow = c(1, 1))
color.vector <- rep(c("deepskyblue", "orange", "forestgreen", "indianred3"), 10)
m = 3098
plot(-log10(myP) ~ seq(1:m), col = color.vector[myGM[, 2]])
abline(v = t(myQTN), lty = 2, lwd = 1, col = "gray")
abline(h = -log10(0.05/m), lty = 1, lwd = 2, col = "black")

