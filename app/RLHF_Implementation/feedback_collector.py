# Feedback Collector for RLHF Training
from typing import Dict, Any, Optional, List, Tuple
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
from pathlib import Path
from app.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

@dataclass
class FeedbackData:
    """Structure for storing feedback information"""
    feedback_id: str
    session_id: str
    agent_name: str
    user_query: str
    agent_response: Dict[str, Any]
    feedback_type: str  # "thumbs", "rating", "comment", "implicit"
    feedback_value: float  # 1.0/-1.0 for thumbs, 1-5 for rating
    timestamp: datetime
    response_time: float
    user_comment: Optional[str] = None
    context_data: Optional[Dict] = None

class FeedbackCollector:
    """
    Collects and processes user feedback for RLHF training
    
    This class handles:
    1. Real-time feedback collection from users
    2. Processing different types of feedback
    3. Tracking agent performance metrics
    4. Preparing data for reward model training
    """
    
    def __init__(self, db_path: str = "data/rlhf_feedback.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Performance tracking
        self.agent_metrics = defaultdict(lambda: {
            'total_queries': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'average_rating': 0.0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'recent_feedback': deque(maxlen=100)
        })
        
        # Feedback processing queue
        self.feedback_queue = deque(maxlen=1000)
        
        # Initialize database
        self._init_database()
        self._load_existing_metrics()
        logger.info(f"Initialized FeedbackCollector with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Feedback table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        feedback_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        user_query TEXT,
                        agent_name TEXT,
                        agent_response TEXT,
                        feedback_type TEXT,
                        feedback_value REAL,
                        timestamp TEXT,
                        response_time REAL,
                        user_comment TEXT,
                        context_data TEXT
                    )
                """)
                
                # Agent performance table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_performance (
                        agent_name TEXT PRIMARY KEY,
                        total_queries INTEGER,
                        positive_feedback INTEGER,
                        negative_feedback INTEGER,
                        average_rating REAL,
                        success_rate REAL,
                        avg_response_time REAL,
                        last_updated TEXT
                    )
                """)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    @log_execution_time
    def collect_feedback(self, 
                        session_id: str,
                        user_query: str,
                        agent_name: str,
                        agent_response: Dict[str, Any],
                        feedback_type: str,
                        feedback_value: float,
                        response_time: float,
                        user_comment: Optional[str] = None,
                        context_data: Optional[Dict] = None) -> str:
        """Collect user feedback for an agent response"""
        
        try:
            # Create feedback data object
            feedback = FeedbackData(
                feedback_id=str(uuid.uuid4()),
                session_id=session_id,
                user_query=user_query,
                agent_name=agent_name,
                agent_response=agent_response,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                timestamp=datetime.now(),
                response_time=response_time,
                user_comment=user_comment,
                context_data=context_data
            )
            
            # Validate feedback
            if not self._validate_feedback(feedback):
                logger.error("Invalid feedback data provided")
                raise ValueError("Invalid feedback data")
            
            # Store feedback
            self._store_feedback(feedback)
            
            # Update agent metrics
            self._update_agent_metrics(feedback)
            
            # Add to processing queue
            self.feedback_queue.append(feedback)
            
            logger.info(f"Collected feedback for {agent_name}: {feedback_type}={feedback_value}")
            return feedback.feedback_id
            
        except Exception as e:
            logger.error(f"Failed to collect feedback: {str(e)}")
            raise

    def _validate_feedback(self, feedback: FeedbackData) -> bool:
        """Validate feedback data before processing"""
        
        # Check required fields
        if not all([feedback.session_id, feedback.user_query, feedback.agent_name]):
            logger.warning("Missing required feedback fields")
            return False
        
        # Validate feedback type and value
        if feedback.feedback_type == "thumbs":
            if feedback.feedback_value not in [-1.0, 1.0]:
                logger.warning(f"Invalid thumbs feedback value: {feedback.feedback_value}")
                return False
        elif feedback.feedback_type == "rating":
            if not (1.0 <= feedback.feedback_value <= 5.0):
                logger.warning(f"Invalid rating feedback value: {feedback.feedback_value}")
                return False
        
        # Check response time is reasonable
        if feedback.response_time < 0 or feedback.response_time > 300:
            logger.warning(f"Unreasonable response time: {feedback.response_time}")
            return False
        
        return True
    
    def _store_feedback(self, feedback: FeedbackData):
        """Store the validated feedback into database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_id,
                    feedback.session_id,
                    feedback.user_query,
                    feedback.agent_name,
                    json.dumps(feedback.agent_response),
                    feedback.feedback_type,
                    feedback.feedback_value,
                    feedback.timestamp.isoformat(),
                    feedback.response_time,
                    feedback.user_comment,
                    json.dumps(feedback.context_data) if feedback.context_data else None
                ))
            logger.debug(f"Stored feedback: {feedback.feedback_id}")
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}")
            raise
    
    def _update_agent_metrics(self, feedback: FeedbackData):
        """Update performance metrics for the agent"""
        try:
            agent_name = feedback.agent_name
            metrics = self.agent_metrics[agent_name]
            
            # Update counters
            metrics['total_queries'] += 1
            
            # Update feedback counts
            if feedback.feedback_value > 0:
                metrics['positive_feedback'] += 1
            else:
                metrics['negative_feedback'] += 1
            
            # Update success rate
            metrics['success_rate'] = metrics['positive_feedback'] / metrics['total_queries']
            
            # Update average rating (for rating-type feedback)
            if feedback.feedback_type == "rating":
                current_avg = metrics['average_rating']
                total_queries = metrics['total_queries']
                metrics['average_rating'] = (current_avg * (total_queries - 1) + feedback.feedback_value) / total_queries
            
            # Update average response time
            current_avg_time = metrics['avg_response_time']
            total_queries = metrics['total_queries']
            metrics['avg_response_time'] = (current_avg_time * (total_queries - 1) + feedback.response_time) / total_queries
            
            # Add to recent feedback
            metrics['recent_feedback'].append(feedback)
            
            # Categorize feedback by task type for advanced analytics
            task_category = self.categorize_by_task_type(feedback)
            if 'task_categories' not in metrics:
                metrics['task_categories'] = defaultdict(int)
            metrics['task_categories'][task_category] += 1
            
            # Store updated metrics in database
            self._store_agent_metrics(agent_name, metrics)
            
            logger.debug(f"Updated metrics for {agent_name}: success_rate={metrics['success_rate']:.2f}, task={task_category}")
            
        except Exception as e:
            logger.error(f"Failed to update agent metrics: {str(e)}")
    
    def _store_agent_metrics(self, agent_name: str, metrics: Dict):
        """Store agent performance metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO agent_performance VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_name,
                    metrics['total_queries'],
                    metrics['positive_feedback'],
                    metrics['negative_feedback'],
                    metrics['average_rating'],
                    metrics['success_rate'],
                    metrics['avg_response_time'],
                    datetime.now().isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to store agent metrics: {str(e)}")
    
    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent"""
        try:
            if agent_name in self.agent_metrics:
                metrics = self.agent_metrics[agent_name].copy()
                # Convert deque to list for JSON serialization
                metrics['recent_feedback'] = [asdict(f) for f in list(metrics['recent_feedback'])]
                logger.info(f"Retrieved performance for {agent_name}")
                return metrics
            logger.warning(f"No performance data found for {agent_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to get agent performance: {str(e)}")
            return None
    
    def get_training_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get processed feedback data for reward model training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT user_query, agent_response, feedback_value, agent_name, response_time
                    FROM feedback 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                training_data = []
                for row in cursor.fetchall():
                    user_query, agent_response, feedback_value, agent_name, response_time = row
                    
                    training_example = {
                        'query': user_query,
                        'response': json.loads(agent_response),
                        'reward': feedback_value,
                        'agent': agent_name,
                        'response_time': response_time
                    }
                    training_data.append(training_example)
                
                logger.info(f"Retrieved {len(training_data)} training examples")
                return training_data
                
        except Exception as e:
            logger.error(f"Failed to get training data: {str(e)}")
            return []
    
    def _load_existing_metrics(self):
        """Load existing agent metrics from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM agent_performance")
                for row in cursor.fetchall():
                    agent_name = row[0]
                    self.agent_metrics[agent_name] = {
                        'total_queries': row[1],
                        'positive_feedback': row[2],
                        'negative_feedback': row[3],
                        'average_rating': row[4],
                        'success_rate': row[5],
                        'avg_response_time': row[6],
                        'recent_feedback': deque(maxlen=100)
                    }
            logger.info("Loaded existing agent metrics from database")
        except sqlite3.OperationalError:
            logger.info("No existing metrics found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load existing metrics: {str(e)}")
    
    def should_retrain_reward_model(self) -> bool:
        """Determine if we have enough new feedback to retrain the reward model"""
        should_retrain = len(self.feedback_queue) >= 100
        if should_retrain:
            logger.info(f"Recommending reward model retraining: {len(self.feedback_queue)} feedback items in queue")
        return should_retrain
    
    def clear_processed_feedback(self):
        """Clear processed feedback from queue after training"""
        cleared_count = len(self.feedback_queue)
        self.feedback_queue.clear()
        logger.info(f"Cleared {cleared_count} processed feedback items from queue")
    
    def export_feedback_data(self, filepath: str, format: str = "json"):
        """Export feedback data for analysis"""
        try:
            training_data = self.get_training_data(limit=10000)
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(training_data, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(training_data)
                df.to_csv(filepath, index=False)
            
            logger.info(f"Exported {len(training_data)} feedback items to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export feedback data: {str(e)}")
            raise
    
    # Advanced Features for NeuroAct Project
    
    def get_agent_comparison(self) -> Dict[str, Any]:
        """Compare performance across all NeuroAct agents"""
        try:
            neuroact_agents = ['DataAgent', 'VisualizationAgent', 'MLAgent', 'CodeAgent', 
                              'NLPAgent', 'DeepLearningAgent', 'ModelEvaluationAgent', 
                              'CriticAgent', 'PlannerAgent', 'RetrievalAgent']
            
            comparison = {}
            for agent_name in neuroact_agents:
                if agent_name in self.agent_metrics:
                    metrics = self.agent_metrics[agent_name]
                    comparison[agent_name] = {
                        'success_rate': round(metrics['success_rate'], 3),
                        'avg_response_time': round(metrics['avg_response_time'], 2),
                        'total_queries': metrics['total_queries'],
                        'average_rating': round(metrics['average_rating'], 2)
                    }
                else:
                    comparison[agent_name] = {'status': 'no_data'}
            
            # Find best and worst performers
            active_agents = {k: v for k, v in comparison.items() if 'success_rate' in v}
            if active_agents:
                best_agent = max(active_agents.items(), key=lambda x: x[1]['success_rate'])
                worst_agent = min(active_agents.items(), key=lambda x: x[1]['success_rate'])
                
                comparison['summary'] = {
                    'best_performer': {'agent': best_agent[0], 'success_rate': best_agent[1]['success_rate']},
                    'worst_performer': {'agent': worst_agent[0], 'success_rate': worst_agent[1]['success_rate']},
                    'total_active_agents': len(active_agents)
                }
            
            logger.info(f"Generated comparison for {len(comparison)} agents")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to generate agent comparison: {str(e)}")
            return {}
    
    def categorize_by_task_type(self, feedback: FeedbackData) -> str:
        """Categorize feedback based on NeuroAct task types"""
        try:
            query = feedback.user_query.lower()
            
            # NeuroAct-specific task categorization
            if any(word in query for word in ['plot', 'chart', 'visualize', 'graph', 'histogram']):
                return 'visualization_task'
            elif any(word in query for word in ['data', 'clean', 'analyze', 'eda', 'outlier']):
                return 'data_processing_task'
            elif any(word in query for word in ['code', 'generate', 'execute', 'python', 'script']):
                return 'code_generation_task'
            elif any(word in query for word in ['train', 'model', 'ml', 'machine learning', 'predict']):
                return 'ml_training_task'
            elif any(word in query for word in ['text', 'nlp', 'sentiment', 'summarize']):
                return 'nlp_task'
            elif any(word in query for word in ['neural', 'deep', 'cnn', 'rnn', 'lstm']):
                return 'deep_learning_task'
            elif any(word in query for word in ['evaluate', 'metrics', 'accuracy', 'optimize']):
                return 'evaluation_task'
            elif any(word in query for word in ['retrieve', 'search', 'rag', 'knowledge']):
                return 'retrieval_task'
            else:
                return 'general_task'
                
        except Exception as e:
            logger.error(f"Failed to categorize task: {str(e)}")
            return 'unknown_task'
    
    def analyze_routing_effectiveness(self) -> Dict[str, Any]:
        """Analyze how well CriticAgent/PlannerAgent routes queries"""
        try:
            routing_stats = {
                'correct_routing': 0,
                'suboptimal_routing': 0,
                'routing_accuracy': 0.0,
                'agent_performance': {}
            }
            
            # Analyze routing effectiveness based on agent success rates
            for agent_name, metrics in self.agent_metrics.items():
                if agent_name in ['CriticAgent', 'PlannerAgent']:
                    continue  # Skip routing agents themselves
                
                total_queries = metrics['total_queries']
                success_rate = metrics['success_rate']
                
                routing_stats['agent_performance'][agent_name] = {
                    'success_rate': round(success_rate, 3),
                    'queries': total_queries,
                    'routing_quality': 'good' if success_rate > 0.7 else 'needs_improvement'
                }
                
                if success_rate > 0.7:  # Good routing
                    routing_stats['correct_routing'] += total_queries
                else:  # Poor routing
                    routing_stats['suboptimal_routing'] += total_queries
            
            total = routing_stats['correct_routing'] + routing_stats['suboptimal_routing']
            if total > 0:
                routing_stats['routing_accuracy'] = round(routing_stats['correct_routing'] / total, 3)
            
            logger.info(f"Routing analysis: {routing_stats['routing_accuracy']:.1%} accuracy")
            return routing_stats
            
        except Exception as e:
            logger.error(f"Failed to analyze routing effectiveness: {str(e)}")
            return {}
    
    def get_quality_training_data(self, min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """Get high-quality feedback for RLHF training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT user_query, agent_response, feedback_value, agent_name, response_time, user_comment
                    FROM feedback 
                    WHERE ABS(feedback_value) >= ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """, (min_confidence,))
                
                quality_data = []
                for row in cursor.fetchall():
                    user_query, agent_response, feedback_value, agent_name, response_time, user_comment = row
                    
                    # Only include clear positive/negative feedback
                    if abs(feedback_value) >= min_confidence:
                        quality_data.append({
                            'query': user_query,
                            'response': json.loads(agent_response),
                            'reward': feedback_value,
                            'agent': agent_name,
                            'confidence': abs(feedback_value),
                            'response_time': response_time,
                            'has_comment': bool(user_comment)
                        })
                
                logger.info(f"Retrieved {len(quality_data)} high-quality training examples")
                return quality_data
                
        except Exception as e:
            logger.error(f"Failed to get quality training data: {str(e)}")
            return []
    
    def track_learning_progress(self, agent_name: str, window_size: int = 50) -> Dict[str, Any]:
        """Track how agent performance improves over time"""
        try:
            if agent_name not in self.agent_metrics:
                return {'status': 'agent_not_found'}
            
            recent_feedback = list(self.agent_metrics[agent_name]['recent_feedback'])
            if len(recent_feedback) < window_size:
                return {'status': 'insufficient_data', 'required': window_size, 'available': len(recent_feedback)}
            
            # Split into early and recent windows
            early_window = recent_feedback[:window_size//2]
            recent_window = recent_feedback[-window_size//2:]
            
            early_success = sum(1 for f in early_window if f.feedback_value > 0) / len(early_window)
            recent_success = sum(1 for f in recent_window if f.feedback_value > 0) / len(recent_window)
            
            # Calculate response time improvement
            early_avg_time = sum(f.response_time for f in early_window) / len(early_window)
            recent_avg_time = sum(f.response_time for f in recent_window) / len(recent_window)
            
            improvement = recent_success - early_success
            time_improvement = early_avg_time - recent_avg_time  # Positive = faster
            
            return {
                'agent_name': agent_name,
                'early_success_rate': round(early_success, 3),
                'recent_success_rate': round(recent_success, 3),
                'success_improvement': round(improvement, 3),
                'early_avg_time': round(early_avg_time, 2),
                'recent_avg_time': round(recent_avg_time, 2),
                'time_improvement': round(time_improvement, 2),
                'is_improving': improvement > 0.05,
                'is_faster': time_improvement > 0.5,
                'learning_trend': 'improving' if improvement > 0.05 else 'stable' if abs(improvement) <= 0.05 else 'declining',
                'overall_progress': 'excellent' if improvement > 0.1 and time_improvement > 1.0 else 
                                  'good' if improvement > 0.05 or time_improvement > 0.5 else 
                                  'needs_attention' if improvement < -0.05 else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Failed to track learning progress: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_feedback_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights from all feedback data"""
        try:
            insights = {
                'agent_comparison': self.get_agent_comparison(),
                'routing_effectiveness': self.analyze_routing_effectiveness(),
                'total_feedback_count': sum(metrics['total_queries'] for metrics in self.agent_metrics.values()),
                'system_health': 'good' if all(metrics['success_rate'] > 0.6 for metrics in self.agent_metrics.values() if metrics['total_queries'] > 10) else 'needs_improvement'
            }
            
            # Add learning progress for top agents
            insights['learning_progress'] = {}
            for agent_name in ['DataAgent', 'VisualizationAgent', 'MLAgent', 'CodeAgent']:
                if agent_name in self.agent_metrics and self.agent_metrics[agent_name]['total_queries'] > 50:
                    progress = self.track_learning_progress(agent_name)
                    if progress.get('status') != 'insufficient_data':
                        insights['learning_progress'][agent_name] = progress
            
            logger.info("Generated comprehensive feedback insights")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate feedback insights: {str(e)}")
            return {}