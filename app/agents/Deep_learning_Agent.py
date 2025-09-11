""" Optimized Deep Learning Agent - Essential features only"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from app.config import get_ollama_config
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from app.utils.data_loader import load_data

logger = get_logger(__name__)

class DeepLearningAgent:
    """ Optimized Deep Learning Agent - ANN, CNN, RNN with key accuracy features"""
    def __init__(self):
        self.name = "DeepLearningAgent"
        self.ollama_config = get_ollama_config()
        self.models = {}
        self.preprocessors = {}
        self._setup_tensorflow()
        logger.info(f"Initialized {self.name}")

    def _setup_tensorflow(self):
        """Setup TensorFlow optimizations once"""
        try:
            import tensorflow as tf
            # Enable mixed precision for performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            # Optimize GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            pass

    def _auto_install(self, package: str) -> bool:
        """Auto-install packages"""
        try:
            import importlib
            importlib.import_module(package)
            return True
        except ImportError:
            try:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                return True
            except:
                return False

    def _auto_import(self, module_path: str, class_name: str):
        """Auto-import with installation"""
        try:
            import importlib
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                packages = {"tensorflow": "tensorflow", "torch": "torch", "transformers": "transformers"}
                pkg = packages.get(module_path.split('.')[0], module_path.split('.')[0])
                if self._auto_install(pkg):
                    module = importlib.import_module(module_path)
                else:
                    return None
            return getattr(module, class_name)
        except:
            return None

    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute deep learning tasks"""
        try:
            query = task.inputs.get("query", "").lower()
            data_path = task.inputs.get("data_path")
            target = task.inputs.get("target_column")
            
            # Intent detection
            if any(word in query for word in ["neural", "nn", "mlp", "deep"]):
                return self._build_neural_network(query, data_path, target)
            elif any(word in query for word in ["cnn", "convolutional", "image"]):
                return self._build_cnn(query, data_path, target)
            elif any(word in query for word in ["rnn", "lstm", "gru", "sequence"]):
                return self._build_rnn(query, data_path, target)
            elif any(word in query for word in ["autoencoder", "encoder"]):
                return self._build_autoencoder(query, data_path)
            elif any(word in query for word in ["annotation", "annotate", "label", "detect", "segment"]):
                return self._do_image_annotation(query, data_path)
            else:
                return self._auto_deep_learning(query, data_path, target)
                
        except Exception as e:
            return {"status": "error", "errors": [str(e)], "agent": self.name}

    def _build_neural_network(self, query: str, data_path: str, target: str = None) -> Dict[str, Any]:
        """Build optimized neural network"""
        try:
            # Load data
            df = load_data(data_path)
            if not target:
                target = df.columns[-1]
            
            X = df.drop(columns=[target])
            y = df[target]
            X_processed = self._preprocess_data(X)
            
            # Task type
            task_type = "classification" if y.dtype == 'object' or len(y.unique()) < 20 else "regression"
            
            # Import layers
            Sequential = self._auto_import("tensorflow.keras.models", "Sequential")
            Dense = self._auto_import("tensorflow.keras.layers", "Dense")
            Dropout = self._auto_import("tensorflow.keras.layers", "Dropout")
            BatchNormalization = self._auto_import("tensorflow.keras.layers", "BatchNormalization")
            
            if not all([Sequential, Dense, Dropout, BatchNormalization]):
                return {"status": "error", "errors": ["Could not import TensorFlow"]}
            
            # Build model with accuracy enhancements
            model = Sequential()
            architecture = self._get_architecture(query, X_processed.shape[1])
            dropout_rate = 0.3 if "dropout" in query else 0.2
            
            # Input layer
            model.add(Dense(architecture[0], activation='relu', input_shape=(X_processed.shape[1],)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            # Hidden layers
            for units in architecture[1:]:
                model.add(Dense(units, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
            
            # Output layer
            if task_type == "classification":
                n_classes = len(y.unique())
                activation = 'softmax' if n_classes > 2 else 'sigmoid'
                model.add(Dense(n_classes if n_classes > 2 else 1, activation=activation))
            else:
                model.add(Dense(1, activation='linear'))
            
            # Advanced compilation
            optimizer = self._get_optimizer(query)
            loss = 'sparse_categorical_crossentropy' if task_type == "classification" else 'mse'
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy' if task_type == "classification" else 'mae'])
            
            # Train with callbacks
            result = self._train_model(model, X_processed, y, query, task_type)
            
            # Save model
            model_id = f"nn_model_{len(self.models)}"
            self.models[model_id] = model
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": "Neural Network",
                "task_type": task_type,
                "architecture": architecture,
                "optimizer": optimizer,
                **result
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _build_cnn(self, query: str, data_path: str, target: str = None) -> Dict[str, Any]:
        """Build optimized CNN"""
        try:
            # Import layers
            Sequential = self._auto_import("tensorflow.keras.models", "Sequential")
            Conv2D = self._auto_import("tensorflow.keras.layers", "Conv2D")
            MaxPooling2D = self._auto_import("tensorflow.keras.layers", "MaxPooling2D")
            Flatten = self._auto_import("tensorflow.keras.layers", "Flatten")
            Dense = self._auto_import("tensorflow.keras.layers", "Dense")
            Dropout = self._auto_import("tensorflow.keras.layers", "Dropout")
            BatchNormalization = self._auto_import("tensorflow.keras.layers", "BatchNormalization")
            
            # Build optimized CNN
            model = Sequential()
            
            # Conv layers with batch norm and dropout
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))
            
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))
            
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))
            
            # Dense layers
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))
            
            # Compile with advanced optimizer
            optimizer = self._get_optimizer(query)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            model_id = f"cnn_model_{len(self.models)}"
            self.models[model_id] = model
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": "Optimized CNN",
                "optimizer": optimizer,
                "message": "CNN with batch norm and dropout created"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _build_rnn(self, query: str, data_path: str, target: str = None) -> Dict[str, Any]:
        """Build optimized RNN"""
        try:
            # Import layers
            Sequential = self._auto_import("tensorflow.keras.models", "Sequential")
            LSTM = self._auto_import("tensorflow.keras.layers", "LSTM")
            GRU = self._auto_import("tensorflow.keras.layers", "GRU")
            Dense = self._auto_import("tensorflow.keras.layers", "Dense")
            Dropout = self._auto_import("tensorflow.keras.layers", "Dropout")
            Bidirectional = self._auto_import("tensorflow.keras.layers", "Bidirectional")
            
            model = Sequential()
            
            # Choose RNN type
            rnn_layer = GRU if "gru" in query else LSTM
            units = 100
            
            # Bidirectional RNN with dropout
            if "bidirectional" in query:
                model.add(Bidirectional(rnn_layer(units, return_sequences=True), input_shape=(None, 1)))
                model.add(Dropout(0.2))
                model.add(Bidirectional(rnn_layer(units//2)))
            else:
                model.add(rnn_layer(units, return_sequences=True, input_shape=(None, 1)))
                model.add(Dropout(0.2))
                model.add(rnn_layer(units//2))
            
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
            # Advanced optimizer
            optimizer = self._get_optimizer(query)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            model_id = f"rnn_model_{len(self.models)}"
            self.models[model_id] = model
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": f"Optimized {'GRU' if 'gru' in query else 'LSTM'}",
                "bidirectional": "bidirectional" in query,
                "optimizer": optimizer
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _build_autoencoder(self, query: str, data_path: str) -> Dict[str, Any]:
        """Build autoencoder"""
        try:
            df = load_data(data_path)
            X_processed = self._preprocess_data(df)
            
            # Import layers
            Model = self._auto_import("tensorflow.keras.models", "Model")
            Input = self._auto_import("tensorflow.keras.layers", "Input")
            Dense = self._auto_import("tensorflow.keras.layers", "Dense")
            
            # Autoencoder architecture
            input_dim = X_processed.shape[1]
            encoding_dim = max(2, input_dim // 4)
            
            # Encoder
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_layer)
            
            # Decoder
            decoded = Dense(input_dim, activation='sigmoid')(encoded)
            
            # Autoencoder
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Train
            autoencoder.fit(X_processed, X_processed, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            
            model_id = f"autoencoder_model_{len(self.models)}"
            self.models[model_id] = autoencoder
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": "Autoencoder",
                "input_dim": input_dim,
                "encoding_dim": encoding_dim
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _train_model(self, model, X, y, query: str, task_type: str) -> Dict[str, Any]:
        """Train model with callbacks"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Encode target if classification
            if task_type == "classification":
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
            
            # Callbacks for accuracy
            EarlyStopping = self._auto_import("tensorflow.keras.callbacks", "EarlyStopping")
            ReduceLROnPlateau = self._auto_import("tensorflow.keras.callbacks", "ReduceLROnPlateau")
            
            callbacks = []
            if EarlyStopping:
                callbacks.append(EarlyStopping(patience=10, restore_best_weights=True))
            if ReduceLROnPlateau:
                callbacks.append(ReduceLROnPlateau(patience=5, factor=0.5))
            
            # Train
            epochs = self._get_epochs(query)
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            test_loss, test_metric = model.evaluate(X_test, y_test, verbose=0)
            
            return {
                "test_loss": float(test_loss),
                "test_metric": float(test_metric),
                "epochs_trained": len(history.history['loss']),
                "final_val_loss": float(history.history['val_loss'][-1])
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _preprocess_data(self, X: pd.DataFrame) -> np.ndarray:
        """Optimized preprocessing"""
        X_processed = X.copy()
        
        # Encode categorical
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean(numeric_only=True)).fillna(0)
        
        # Scale for deep learning
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(X_processed)

    def _get_architecture(self, query: str, input_dim: int) -> List[int]:
        """Get optimal architecture"""
        import re
        numbers = re.findall(r'\d+', query)
        if numbers:
            return [int(n) for n in numbers[:3]]
        
        # Auto-size based on input
        if input_dim < 10:
            return [64, 32]
        elif input_dim < 50:
            return [128, 64, 32]
        else:
            return [256, 128, 64]

    def _get_optimizer(self, query: str):
        """Get advanced optimizer"""
        import tensorflow as tf
        
        if "adamw" in query:
            return tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
        elif "nadam" in query:
            return tf.keras.optimizers.Nadam(learning_rate=0.001)
        elif "sgd" in query:
            return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        else:
            return tf.keras.optimizers.Adam(learning_rate=0.001)

    def _get_epochs(self, query: str) -> int:
        """Extract epochs"""
        import re
        if "epoch" in query:
            numbers = re.findall(r'(\d+)\s*epoch', query)
            if numbers:
                return int(numbers[0])
        return 100

    def _do_image_annotation(self, query: str, image_path: str) -> Dict[str, Any]:
        """Perform image annotation based on query"""
        try:
            # Auto-install OpenCV
            import cv2
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"status": "error", "errors": ["Could not load image"]}
            
            annotations = []
            
            # Face Detection
            if any(word in query.lower() for word in ["face", "person"]):
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    annotations.append({
                        "type": "face",
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "confidence": 0.8
                    })
            
            # Edge Detection
            elif any(word in query.lower() for word in ["edge", "contour"]):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for i, contour in enumerate(contours[:5]):
                    area = cv2.contourArea(contour)
                    if area > 100:
                        x, y, w, h = cv2.boundingRect(contour)
                        annotations.append({
                            "type": "edge",
                            "area": float(area),
                            "bbox": [int(x), int(y), int(w), int(h)]
                        })
            
            # Object Detection (simple)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for i, contour in enumerate(contours[:3]):
                    area = cv2.contourArea(contour)
                    if area > 1000:
                        x, y, w, h = cv2.boundingRect(contour)
                        annotations.append({
                            "type": "object",
                            "area": float(area),
                            "bbox": [int(x), int(y), int(w), int(h)]
                        })
            
            # Draw annotations
            annotated = image.copy()
            for ann in annotations:
                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    color = (0, 255, 0) if ann['type'] == 'face' else (255, 0, 0)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(annotated, ann['type'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save annotated image
            output_path = image_path.replace('.', '_annotated.')
            cv2.imwrite(output_path, annotated)
            
            return {
                "status": "success",
                "annotation_type": ann['type'] if annotations else "general",
                "annotations_found": len(annotations),
                "annotations": annotations,
                "output_image": output_path,
                "message": f"Image annotated with {len(annotations)} annotations"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _auto_deep_learning(self, query: str, data_path: str, target: str = None) -> Dict[str, Any]:
        """Auto-select approach"""
        try:
            if data_path:
                df = load_data(data_path)
                if len(df.columns) > 100:
                    return self._build_autoencoder(query, data_path)
                elif target and target in df.columns:
                    return self._build_neural_network(query, data_path, target)
                else:
                    return self._build_autoencoder(query, data_path)
            else:
                return {"status": "success", "message": "Deep Learning Agent ready"}
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}