""" NLP Agent - Specialized for Natural Language Processing"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from app.config import get_ollama_config
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from app.utils.data_loader import load_data

logger = get_logger(__name__)

class NLPAgent:
    """ NLP Agent - Embeddings, Sentiment, NER, Summarization, Translation"""
    def __init__(self):
        self.name = "NLPAgent"
        self.ollama_config = get_ollama_config()
        self.models = {}
        self.embeddings = {}
        logger.info(f"Initialized {self.name}")

    def _auto_install(self, package: str) -> bool:
        """Auto-install NLP packages"""
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
    
    def _auto_import(self, module_path: str, class_name: str = ""):
        """Auto-import NLP libraries"""
        try:
            import importlib
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                packages = {
                    "nltk": "nltk",
                    "spacy": "spacy", 
                    "transformers": "transformers",
                    "gensim": "gensim",
                    "textblob": "textblob"
                }
                pkg = packages.get(module_path.split('.')[0], module_path.split('.')[0])
                if self._auto_install(pkg):
                    module = importlib.import_module(module_path)
                else:
                    return None
            return getattr(module, class_name) if class_name else module
        except:
            return None
    
    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute NLP tasks"""
        try:
            query = task.inputs.get("query", "").lower()
            text_data = task.inputs.get("text_data", "")
            data_path = task.inputs.get("data_path")
            
            # NLP intent detection
            if any(word in query for word in ["embedding", "word2vec", "bert"]):
                return self._create_embeddings(query, text_data, data_path)
            elif any(word in query for word in ["sentiment", "emotion"]):
                return self._analyze_sentiment(query, text_data, data_path)
            elif any(word in query for word in ["ner", "entity", "extract"]):
                return self._extract_entities(query, text_data, data_path)
            elif any(word in query for word in ["summarize", "summary"]):
                return self._summarize_text(query, text_data, data_path)
            elif any(word in query for word in ["classify", "classification"]):
                return self._classify_text(query, text_data, data_path)
            elif any(word in query for word in ["topic", "lda"]):
                return self._topic_modeling(query, text_data, data_path)
            elif any(word in query for word in ["similarity", "compare"]):
                return self._text_similarity(query, text_data, data_path)
            else:
                return self._auto_nlp(query, text_data, data_path)
                
        except Exception as e:
            return {"status": "error", "errors": [str(e)], "agent": self.name}

    def _create_embeddings(self, query: str, text_data: str, data_path: str = None) -> Dict[str, Any]:
        """Create word embeddings"""
        try:
            texts = self._load_text_data(text_data, data_path)
            if not texts:
                return {"status": "error", "errors": ["No text data provided"]}
            
            if "word2vec" in query:
                return self._create_word2vec(texts, query)
            elif "bert" in query:
                return self._create_bert_embeddings(texts, query)
            else:
                return self._create_word2vec(texts, query)
                
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _create_word2vec(self, texts: List[str], query: str) -> Dict[str, Any]:
        """Create Word2Vec embeddings"""
        try:
            Word2Vec = self._auto_import("gensim.models", "Word2Vec")
            if not Word2Vec:
                return {"status": "error", "errors": ["Could not import Word2Vec"]}
            
            processed_texts = self._preprocess_texts(texts)
            vector_size = self._extract_number(query, 100)
            
            model = Word2Vec(
                sentences=processed_texts,
                vector_size=vector_size,
                window=5,
                min_count=1,
                workers=4
            )
            
            model_id = f"word2vec_model_{len(self.models)}"
            self.models[model_id] = model
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": "Word2Vec",
                "vector_size": vector_size,
                "vocab_size": len(model.wv.key_to_index),
                "message": f"Word2Vec model trained with {len(model.wv.key_to_index)} words"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _create_bert_embeddings(self, texts: List[str], query: str) -> Dict[str, Any]:
        """Create BERT embeddings"""
        try:
            AutoTokenizer = self._auto_import("transformers", "AutoTokenizer")
            AutoModel = self._auto_import("transformers", "AutoModel")
            
            if not all([AutoTokenizer, AutoModel]):
                return {"status": "error", "errors": ["Could not import transformers"]}
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")
            
            embeddings = []
            for text in texts[:50]:  # Limit for performance
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                embeddings.append(embedding[0])
            
            model_id = f"bert_model_{len(self.models)}"
            self.models[model_id] = {"model": model, "tokenizer": tokenizer}
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": "BERT",
                "embedding_dim": 768,
                "texts_processed": len(embeddings),
                "message": f"BERT embeddings created for {len(embeddings)} texts"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _analyze_sentiment(self, query: str, text_data: str, data_path: str = None) -> Dict[str, Any]:
        """Analyze sentiment"""
        try:
            texts = self._load_text_data(text_data, data_path)
            if not texts:
                return {"status": "error", "errors": ["No text data provided"]}
            
            TextBlob = self._auto_import("textblob", "TextBlob")
            if not TextBlob:
                return {"status": "error", "errors": ["Could not import TextBlob"]}
            
            results = []
            for text in texts:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment = "positive"
                elif polarity < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": sentiment,
                    "polarity": round(polarity, 3)
                })
            
            sentiment_counts = {
                "positive": sum(1 for r in results if r["sentiment"] == "positive"),
                "negative": sum(1 for r in results if r["sentiment"] == "negative"),
                "neutral": sum(1 for r in results if r["sentiment"] == "neutral")
            }
            
            return {
                "status": "success",
                "analysis_type": "Sentiment Analysis",
                "total_texts": len(texts),
                "sentiment_distribution": sentiment_counts,
                "results": results[:20],
                "average_polarity": round(np.mean([r["polarity"] for r in results]), 3)
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _extract_entities(self, query: str, text_data: str, data_path: str = None) -> Dict[str, Any]:
        """Extract named entities"""
        try:
            texts = self._load_text_data(text_data, data_path)
            if not texts:
                return {"status": "error", "errors": ["No text data provided"]}
            
            spacy = self._auto_import("spacy")
            if not spacy:
                return {"status": "error", "errors": ["Could not import spaCy"]}
            
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm")
            
            all_entities = []
            entity_counts = {}
            
            for text in texts:
                doc = nlp(text)
                for ent in doc.ents:
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "description": spacy.explain(ent.label_)
                    }
                    all_entities.append(entity_info)
                    entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
            
            return {
                "status": "success",
                "analysis_type": "Named Entity Recognition",
                "total_entities": len(all_entities),
                "entity_distribution": entity_counts,
                "entities": all_entities[:30],
                "message": f"Extracted {len(all_entities)} entities from {len(texts)} texts"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _summarize_text(self, query: str, text_data: str, data_path: str = None) -> Dict[str, Any]:
        """Summarize text"""
        try:
            texts = self._load_text_data(text_data, data_path)
            if not texts:
                return {"status": "error", "errors": ["No text data provided"]}
            
            pipeline = self._auto_import("transformers", "pipeline")
            if not pipeline:
                return {"status": "error", "errors": ["Could not import transformers"]}
            
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            summaries = []
            for text in texts[:3]:  # Limit for performance
                if len(text) > 100:
                    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
                    summaries.append({
                        "original_text": text[:200] + "..." if len(text) > 200 else text,
                        "summary": summary[0]["summary_text"]
                    })
            
            return {
                "status": "success",
                "analysis_type": "Text Summarization",
                "texts_summarized": len(summaries),
                "summaries": summaries,
                "message": f"Summarized {len(summaries)} texts"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _classify_text(self, query: str, text_data: str, data_path: str = None) -> Dict[str, Any]:
        """Classify text"""
        try:
            texts = self._load_text_data(text_data, data_path)
            if not texts:
                return {"status": "error", "errors": ["No text data provided"]}
            
            pipeline = self._auto_import("transformers", "pipeline")
            if not pipeline:
                return {"status": "error", "errors": ["Could not import transformers"]}
            
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            categories = ["positive", "negative", "neutral", "business", "technology"]
            
            results = []
            for text in texts[:5]:  # Limit for performance
                result = classifier(text, categories)
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "predicted_category": result["labels"][0],
                    "confidence": round(result["scores"][0], 3)
                })
            
            return {
                "status": "success",
                "analysis_type": "Text Classification",
                "categories": categories,
                "texts_classified": len(results),
                "results": results,
                "message": f"Classified {len(results)} texts"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _topic_modeling(self, query: str, text_data: str, data_path: str = None) -> Dict[str, Any]:
        """Topic modeling"""
        try:
            texts = self._load_text_data(text_data, data_path)
            if not texts:
                return {"status": "error", "errors": ["No text data provided"]}
            
            LdaModel = self._auto_import("gensim.models", "LdaModel")
            Dictionary = self._auto_import("gensim.corpora", "Dictionary")
            
            if not all([LdaModel, Dictionary]):
                return {"status": "error", "errors": ["Could not import Gensim"]}
            
            processed_texts = self._preprocess_texts(texts)
            dictionary = Dictionary(processed_texts)
            corpus = [dictionary.doc2bow(text) for text in processed_texts]
            
            num_topics = self._extract_number(query, 5)
            
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10
            )
            
            topics = []
            for idx, topic in lda_model.print_topics(-1):
                topics.append({
                    "topic_id": idx,
                    "words": topic
                })
            
            model_id = f"lda_model_{len(self.models)}"
            self.models[model_id] = lda_model
            
            return {
                "status": "success",
                "model_id": model_id,
                "analysis_type": "Topic Modeling",
                "num_topics": num_topics,
                "topics": topics,
                "message": f"Discovered {num_topics} topics"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _text_similarity(self, query: str, text_data: str, data_path: str = None) -> Dict[str, Any]:
        """Text similarity"""
        try:
            texts = self._load_text_data(text_data, data_path)
            if len(texts) < 2:
                return {"status": "error", "errors": ["Need at least 2 texts"]}
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            similar_pairs = []
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    similarity = similarity_matrix[i][j]
                    if similarity > 0.3:
                        similar_pairs.append({
                            "text1": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                            "text2": texts[j][:100] + "..." if len(texts[j]) > 100 else texts[j],
                            "similarity": round(similarity, 3)
                        })
            
            similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "status": "success",
                "analysis_type": "Text Similarity",
                "total_texts": len(texts),
                "similar_pairs": similar_pairs[:10],
                "average_similarity": round(np.mean(similarity_matrix), 3)
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _load_text_data(self, text_data: str, data_path: str = None) -> List[str]:
        """Load text data"""
        texts = []
        
        if text_data:
            texts.append(text_data)
        
        if data_path:
            try:
                if data_path.endswith('.csv'):
                    df = pd.read_csv(data_path)
                    text_col = df.select_dtypes(include=['object']).columns[0]
                    texts.extend(df[text_col].dropna().tolist())
                elif data_path.endswith('.txt'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        texts.extend(f.readlines())
            except:
                pass
        
        return [str(text).strip() for text in texts if text and str(text).strip()]

    def _preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """Preprocess texts"""
        processed = []
        for text in texts:
            words = text.lower().split()
            words = [word.strip('.,!?";\'()[]{}') for word in words if len(word) > 2]
            processed.append(words)
        return processed

    def _extract_number(self, query: str, default: int) -> int:
        """Extract number from query"""
        import re
        numbers = re.findall(r'\d+', query)
        return int(numbers[0]) if numbers else default

    def _auto_nlp(self, query: str, text_data: str, data_path: str = None) -> Dict[str, Any]:
        """Auto NLP task"""
        return self._analyze_sentiment(query, text_data, data_path)