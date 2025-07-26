from rag_pipeline import RAGPipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGEvaluator:
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        
        # Sample evaluation data (same as notebook)
        self.evaluation_data = [
            {"query": "অনুপমের বন্ধুর নাম কি?", "ground_truth": "হরিশ"},
            {"query": "কল্যাণীর পিতার নাম কি?", "ground_truth": "শস্তুনাথ সেন"},
            {"query": "অনুপমের বাবা কি করে জীবিকা নির্বাহ করতেন?", "ground_truth": "ওকালতি "},
            {"query": "মামাকে ভাগ্য দেবতার প্রধান এজেন্ট বলা হয়েছে কেন ?", "ground_truth": "তার প্রভাব "},
            {"query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে ?", "ground_truth": "মামাকে"},
            {"query": "অপরিচিতা গল্পের লেখক কে?", "ground_truth": "রবীন্দ্রনাথ ঠাকুর"},
        ]
        
        # Extended evaluation queries for cosine similarity analysis
        self.similarity_test_queries = [
            # Bengali queries
            {"query": "অনুপমের বন্ধুর নাম কি?", "expected_keywords": ["অনুপম", "বন্ধু", "হরিশ"], "language": "bengali"},
            {"query": "অনুপমের শিক্ষাগত যোগ্যতা কি?", "expected_keywords": ["অনুপম", "শিক্ষা", "বি.এ"], "language": "bengali"},
            {"query": "অপরিচিত গল্পের লেখক কে?", "expected_keywords": ["অপরিচিত", "লেখক", "রবীন্দ্রনাথ"], "language": "bengali"},
            {"query": "কল্যাণীর পিতার নাম কি?", "expected_keywords": ["কল্যাণী", "পিতা", "নাম"], "language": "bengali"},
            {"query": "মামা কোন ঘরের মেয়ে পছন্দ করতেন?", "expected_keywords": ["মামা", "ঘর", "মেয়ে", "পছন্দ"], "language": "bengali"},
            
            # English queries
            {"query": "Who is the main character?", "expected_keywords": ["character", "main", "protagonist"], "language": "english"},
            {"query": "What is the central theme?", "expected_keywords": ["theme", "central", "main"], "language": "english"},
            {"query": "Who is Anupam's friend?", "expected_keywords": ["Anupam", "friend", "Harish"], "language": "english"},
            {"query": "What is the author's name?", "expected_keywords": ["author", "name", "writer"], "language": "english"},
        ]
    
    def load_pdf(self, pdf_path):
        """Load and process PDF for evaluation"""
        self.rag_pipeline.load_and_process_pdf(pdf_path)
    
    def calculate_cosine_similarities(self, query, top_k=5):
        """Calculate and return cosine similarities for a query"""
        if self.rag_pipeline.vector_database is None:
            return None
        
        # Get query embedding
        query_embedding = self.rag_pipeline.embedding_generator.generate_query_embedding(query)
        
        # Calculate similarities with all chunks
        similarities = []
        for i, entry in enumerate(self.rag_pipeline.vector_database):
            chunk_text = entry["text"]
            chunk_embedding = entry["embedding"]
            
            # Calculate cosine similarity
            similarity_score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            
            similarities.append({
                "chunk_index": i,
                "chunk_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "similarity_score": similarity_score,
                "full_chunk": chunk_text
            })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return similarities[:top_k]
    
    def analyze_similarity_distribution(self, query):
        """Analyze the distribution of similarity scores for a query"""
        if self.rag_pipeline.vector_database is None:
            return None
        
        # Get all similarities
        all_similarities = self.calculate_cosine_similarities(query, top_k=len(self.rag_pipeline.vector_database))
        
        if not all_similarities:
            return None
        
        scores = [item["similarity_score"] for item in all_similarities]
        
        analysis = {
            "query": query,
            "total_chunks": len(scores),
            "max_similarity": max(scores),
            "min_similarity": min(scores),
            "mean_similarity": np.mean(scores),
            "std_similarity": np.std(scores),
            "median_similarity": np.median(scores),
            "top_10_percent_threshold": np.percentile(scores, 90),
            "highly_relevant_chunks": len([s for s in scores if s > 0.7]),
            "moderately_relevant_chunks": len([s for s in scores if 0.5 <= s <= 0.7]),
            "low_relevance_chunks": len([s for s in scores if s < 0.5])
        }
        
        return analysis
    
    def evaluate_with_similarities(self):
        """Enhanced evaluation including cosine similarity analysis"""
        if self.rag_pipeline.vector_database is None:
            print("❌ Please load a PDF first using load_pdf() method")
            return
        
        print("🔍 Running Enhanced Evaluation with Cosine Similarity Analysis")
        print("=" * 70)
        
        detailed_results = []
        
        for item in self.similarity_test_queries:
            query = item["query"]
            expected_keywords = item["expected_keywords"]
            language = item["language"]
            
            print(f"\n📋 Query: {query}")
            print(f"🏷️  Language: {language.capitalize()}")
            print(f"🔑 Expected Keywords: {expected_keywords}")
            
            # Get top similarities
            similarities = self.calculate_cosine_similarities(query, top_k=3)
            
            if similarities:
                print(f"\n📊 Top 3 Similar Chunks:")
                for i, sim in enumerate(similarities, 1):
                    print(f"   {i}. Score: {sim['similarity_score']:.4f}")
                    print(f"      Text: {sim['chunk_text']}")
                    print()
                
                # Generate answer
                generated_answer = self.rag_pipeline.query(query)
                print(f"🤖 Generated Answer: {generated_answer}")
                
                # Analyze similarity distribution
                distribution = self.analyze_similarity_distribution(query)
                print(f"📈 Similarity Analysis:")
                print(f"   • Max Score: {distribution['max_similarity']:.4f}")
                print(f"   • Mean Score: {distribution['mean_similarity']:.4f}")
                print(f"   • Highly Relevant (>0.7): {distribution['highly_relevant_chunks']} chunks")
                print(f"   • Moderately Relevant (0.5-0.7): {distribution['moderately_relevant_chunks']} chunks")
                
                detailed_results.append({
                    "query": query,
                    "language": language,
                    "expected_keywords": expected_keywords,
                    "top_similarities": similarities,
                    "generated_answer": generated_answer,
                    "similarity_distribution": distribution
                })
            
            print("-" * 70)
        
        return detailed_results
    
    def similarity_benchmark(self):
        """Benchmark similarity scores across different query types"""
        if self.rag_pipeline.vector_database is None:
            print("❌ Please load a PDF first")
            return
        
        print("🏆 Cosine Similarity Benchmark Analysis")
        print("=" * 50)
        
        benchmark_results = {
            "bengali_queries": [],
            "english_queries": [],
            "overall_stats": {}
        }
        
        all_scores = []
        
        for item in self.similarity_test_queries:
            query = item["query"]
            language = item["language"]
            
            # Get similarity analysis
            distribution = self.analyze_similarity_distribution(query)
            
            if distribution:
                query_result = {
                    "query": query,
                    "max_similarity": distribution["max_similarity"],
                    "mean_similarity": distribution["mean_similarity"],
                    "highly_relevant_count": distribution["highly_relevant_chunks"]
                }
                
                if language == "bengali":
                    benchmark_results["bengali_queries"].append(query_result)
                else:
                    benchmark_results["english_queries"].append(query_result)
                
                all_scores.extend([distribution["max_similarity"], distribution["mean_similarity"]])
        
        # Calculate overall statistics
        if all_scores:
            benchmark_results["overall_stats"] = {
                "total_queries_tested": len(self.similarity_test_queries),
                "average_max_similarity": np.mean([q["max_similarity"] for q in benchmark_results["bengali_queries"] + benchmark_results["english_queries"]]),
                "average_mean_similarity": np.mean([q["mean_similarity"] for q in benchmark_results["bengali_queries"] + benchmark_results["english_queries"]]),
                "bengali_avg_max": np.mean([q["max_similarity"] for q in benchmark_results["bengali_queries"]]) if benchmark_results["bengali_queries"] else 0,
                "english_avg_max": np.mean([q["max_similarity"] for q in benchmark_results["english_queries"]]) if benchmark_results["english_queries"] else 0
            }
        
        # Print results
        print(f"📊 Overall Performance:")
        stats = benchmark_results["overall_stats"]
        print(f"   • Average Max Similarity: {stats['average_max_similarity']:.4f}")
        print(f"   • Average Mean Similarity: {stats['average_mean_similarity']:.4f}")
        print(f"   • Bengali Avg Max: {stats['bengali_avg_max']:.4f}")
        print(f"   • English Avg Max: {stats['english_avg_max']:.4f}")
        
        print(f"\n🇧🇩 Bengali Query Performance:")
        for q in benchmark_results["bengali_queries"]:
            print(f"   • {q['query'][:50]}... → Max: {q['max_similarity']:.4f}")
        
        print(f"\n🇺🇸 English Query Performance:")
        for q in benchmark_results["english_queries"]:
            print(f"   • {q['query'][:50]}... → Max: {q['max_similarity']:.4f}")
        
        return benchmark_results
    
    def evaluate(self):
        """Original evaluation method (maintained for compatibility)"""
        if self.rag_pipeline.vector_database is None:
            print("❌ Please load a PDF first using load_pdf() method")
            return
        
        evaluation_results = []
        
        for item in self.evaluation_data:
            query = item["query"]
            ground_truth = item["ground_truth"]
            generated_answer = self.rag_pipeline.query(query)
            
            # Qualitative assessment (manual inspection)
            assessment = "Correct" if generated_answer.strip().lower() == ground_truth.strip().lower() else "Needs Review"
            
            evaluation_results.append({
                "query": query,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "assessment": assessment
            })
        
        return evaluation_results
    
    def comprehensive_evaluation(self):
        """Run all evaluation methods"""
        print("🚀 Starting Comprehensive RAG Evaluation")
        print("=" * 60)
        
        # Basic evaluation
        print("\n1️⃣ Basic Evaluation:")
        basic_results = self.evaluate()
        self.print_evaluation_results(basic_results)
        
        # Similarity-based evaluation
        print("\n2️⃣ Cosine Similarity Evaluation:")
        similarity_results = self.evaluate_with_similarities()
        
        # Benchmark analysis
        print("\n3️⃣ Similarity Benchmark:")
        benchmark_results = self.similarity_benchmark()
        
        # Sample queries test
        print("\n4️⃣ Sample Queries Test:")
        self.test_sample_queries()
        
        return {
            "basic_evaluation": basic_results,
            "similarity_evaluation": similarity_results,
            "benchmark_results": benchmark_results
        }
    
    def print_evaluation_results(self, results):
        """Print evaluation results in a formatted way"""
        for result in results:
            print(f"Query: {result['query']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Generated Answer: {result['generated_answer']}")
            print(f"Assessment: {result['assessment']}")
            print("-" * 50)
    
    def test_sample_queries(self):
        """Test with sample queries from the notebook"""
        sample_queries = [
            'অনুপমের বন্ধুর নাম কি?',
            'অনুপমের শিক্ষাগত যোগ্যতা কি?',
            'অনুপমের বাবা কি করে জীবিকা নির্বাহ করতেন?',
            'আসর জমাতে অদ্বিতীয় কে?',
            'মামা কোন ঘরের মেয়ে পছন্দ করতেন?',
            'অপরিচিত গল্পের লেখক কে?',
            'কল্যাণীর পিতার নাম কি?',
            'রসনচৌকি শব্দের অর্থ কি?'
        ]
        
        print("🧪 Testing Sample Queries with Similarity Scores:")
        print("=" * 55)
        
        for query in sample_queries:
            print(f"\n📝 Query: {query}")
            
            # Get top similarities
            similarities = self.calculate_cosine_similarities(query, top_k=2)
            if similarities:
                print(f"🎯 Top Similarity Scores:")
                for i, sim in enumerate(similarities, 1):
                    print(f"   {i}. {sim['similarity_score']:.4f} - {sim['chunk_text'][:100]}...")
            
            # Get answer
            answer = self.rag_pipeline.query(query)
            print(f"🤖 Answer: {answer}")
            print("-" * 50)
    
    def export_similarity_analysis(self, filename="similarity_analysis.txt"):
        """Export detailed similarity analysis to file"""
        if self.rag_pipeline.vector_database is None:
            print("❌ Please load a PDF first")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Cosine Similarity Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            for item in self.similarity_test_queries:
                query = item["query"]
                f.write(f"Query: {query}\n")
                f.write(f"Language: {item['language']}\n")
                
                # Get similarities
                similarities = self.calculate_cosine_similarities(query, top_k=5)
                if similarities:
                    f.write("Top 5 Similar Chunks:\n")
                    for i, sim in enumerate(similarities, 1):
                        f.write(f"  {i}. Score: {sim['similarity_score']:.4f}\n")
                        f.write(f"     Text: {sim['full_chunk']}\n\n")
                
                # Get distribution analysis
                distribution = self.analyze_similarity_distribution(query)
                if distribution:
                    f.write(f"Distribution Analysis:\n")
                    f.write(f"  Max: {distribution['max_similarity']:.4f}\n")
                    f.write(f"  Mean: {distribution['mean_similarity']:.4f}\n")
                    f.write(f"  Std: {distribution['std_similarity']:.4f}\n")
                    f.write(f"  Highly Relevant (>0.7): {distribution['highly_relevant_chunks']}\n")
                    f.write(f"  Moderately Relevant (0.5-0.7): {distribution['moderately_relevant_chunks']}\n\n")
                
                f.write("-" * 60 + "\n\n")
        
        print(f"✅ Similarity analysis exported to {filename}")

if __name__ == "__main__":
    # Example usage
    evaluator = RAGEvaluator()
    
    # Load your PDF (replace with actual path)
    evaluator.load_pdf("/media/rhythm/3a70cfe7-b9b6-4d81-bb67-b4665d378b3a/rhythm/Code/multilingual-rag-chatbot/HSC26-Bangla1st-Paper.pdf")
    
    # Run comprehensive evaluation with similarity analysis
    results = evaluator.comprehensive_evaluation()
    
    # Export detailed analysis
    evaluator.export_similarity_analysis()
    
    print("Enhanced Evaluator initialized with Cosine Similarity Analysis!")
    print("\nAvailable methods:")
    print("• load_pdf(pdf_path) - Load document")
    print("• comprehensive_evaluation() - Run all evaluations")
    print("• evaluate_with_similarities() - Detailed similarity analysis")
    print("• similarity_benchmark() - Compare performance across languages")
    print("• export_similarity_analysis() - Export detailed report")