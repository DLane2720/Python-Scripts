#!/usr/bin/env python3

"""
Adaptive Paper Analyzer
----------------------

This script provides an adaptive analysis of academic papers using local LLM models via Ollama.
It performs a multi-stage analysis to extract and synthesize information from academic papers,
adapting its approach based on the paper's content and domain.

Requirements:
------------
- Python 3.8+
- Ollama installed and running locally
- Required Python packages:
    pip install pyyaml requests PyPDF2 rich

Setup:
------
1. Ensure Ollama is installed and running
2. Install required Python packages
3. Place both analyze_paper.py and adaptive_paper_analyzer.yaml in the same directory
4. Verify you have at least one model installed in Ollama (e.g., llama3.1:8b)

Usage:
------
Basic usage:
    python analyze_paper.py path/to/paper.pdf

With optional parameters:
    python analyze_paper.py path/to/paper.pdf --output analysis.md --model llama3.1:8b --config custom_config.yaml

Arguments:
    pdf_path        Path to the PDF file to analyze (required)
    --config        Path to YAML configuration file (default: adaptive_paper_analyzer.yaml)
    --model         Name of the Ollama model to use (default: llama3.1:8b)
    --output        Output file path for the analysis (optional, defaults to console output)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

class AdaptivePaperAnalyzer:
    def __init__(self, config_path: str, model_name: str = "llama3.1:8b"):
        """
        Initialize the adaptive paper analyzer.
        
        Args:
            config_path: Path to YAML configuration file
            model_name: Name of the Ollama model to use
        """
        self.console = Console()
        self.setup_logging()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            sys.exit(1)
            
        self.model_name = model_name
        self.validate_ollama_connection()

    def setup_logging(self) -> None:
        """Configure rich logging for the analyzer."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.logger = logging.getLogger("paper_analyzer")

    def validate_ollama_connection(self) -> None:
        """Verify that Ollama is accessible and the model is available."""
        try:
            response = requests.get('http://localhost:11434/api/tags')
            if response.status_code != 200:
                self.logger.error("Failed to connect to Ollama service")
                sys.exit(1)
                
            models = response.json()
            if not any(self.model_name in model['name'] for model in models['models']):
                self.logger.warning(f"Model {self.model_name} not found in available models")
                self.logger.info("Available models:")
                for model in models['models']:
                    self.logger.info(f"  - {model['name']}")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            self.logger.error("Could not connect to Ollama service. Is it running?")
            sys.exit(1)

    def query_ollama(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """
        Send a query to Ollama and get the response.
        
        Args:
            prompt: The prompt to send to Ollama
            temperature: Controls randomness in the response (0.0 to 1.0)
            
        Returns:
            The model's response or None if there's an error
        """
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error response from Ollama: {response.status_code}")
                self.logger.error(f"Response content: {response.text}")
                return None
                
            return response.json().get('response', '')
            
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {str(e)}")
            return None

    def analyze_domain(self, paper_content: str) -> Dict:
        """
        First analysis stage: Identify paper's domain and key concepts.
        """
        self.logger.info("Starting domain analysis...")
        
        prompt = f"""
        {self.config['analysis_framework']['domain_analysis']['instructions']}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "primary_fields": ["field1", "field2"],
            "contribution_type": "type of paper",
            "key_terminology": ["term1", "term2"],
            "major_concepts": ["concept1", "concept2"]
        }}
        
        Paper content:
        {paper_content[:4000]}
        """
        
        response = self.query_ollama(prompt, temperature=0.3)
        
        # Try to extract JSON from the response if it's embedded in other text
        try:
            # Look for JSON-like structure
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            self.logger.warning("Could not parse response as JSON, using formatted dict instead")
            # Create a structured dict from the response even if JSON parsing fails
            lines = response.split('\n')
            result = {
                "primary_fields": [],
                "contribution_type": "unknown",
                "key_terminology": [],
                "major_concepts": []
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.lower().startswith('primary fields'):
                    current_section = "primary_fields"
                elif line.lower().startswith('contribution type'):
                    current_section = "contribution_type"
                elif line.lower().startswith('key terminology'):
                    current_section = "key_terminology"
                elif line.lower().startswith('major concepts'):
                    current_section = "major_concepts"
                elif line and current_section:
                    if current_section == "contribution_type":
                        result[current_section] = line
                    else:
                        # Remove bullet points and other common markers
                        clean_line = line.lstrip('•-*> ')
                        if clean_line:
                            result[current_section].append(clean_line)
                            
            return result

    def map_knowledge_structure(self, paper_content: str, domain_analysis: Dict) -> Dict:
        """
        Second analysis stage: Map the hierarchical knowledge structure.
        """
        self.logger.info("Mapping knowledge structure...")
        
        concepts = json.dumps(domain_analysis['major_concepts'], indent=2)
        prompt = f"""
        {self.config['analysis_framework']['structural_analysis']['instructions']}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "concept_hierarchies": ["concept1", "concept2"],
            "relationships": ["relationship1", "relationship2"],
            "depth_analysis": {{"concept1": "depth"}},
            "novel_contributions": ["contribution1", "contribution2"]
        }}
        
        Based on these concepts:
        {concepts}
        
        Paper content:
        {paper_content[:4000]}
        """
        
        response = self.query_ollama(prompt, temperature=0.3)
        
        try:
            # Look for JSON-like structure
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            self.logger.warning("Could not parse response as JSON, using formatted dict instead")
            # Create a structured dict from the response
            result = {
                "concept_hierarchies": [],
                "relationships": [],
                "depth_analysis": {},
                "novel_contributions": []
            }
            
            current_section = None
            for line in response.split('\n'):
                line = line.strip()
                if 'concept hierarchies' in line.lower():
                    current_section = "concept_hierarchies"
                elif 'relationships' in line.lower():
                    current_section = "relationships"
                elif 'depth analysis' in line.lower():
                    current_section = "depth_analysis"
                elif 'novel contributions' in line.lower():
                    current_section = "novel_contributions"
                elif line and current_section:
                    # Remove bullet points and other common markers
                    clean_line = line.lstrip('•-*> ')
                    if clean_line:
                        if current_section == "depth_analysis" and ':' in clean_line:
                            key, value = clean_line.split(':', 1)
                            result[current_section][key.strip()] = value.strip()
                        else:
                            result[current_section].append(clean_line)
            
            return result

    def _determine_analysis_type(self, concept: str) -> str:
        """
        Helper method to determine the type of analysis needed for a concept.
        """
        # Handle case where concept is a list
        if isinstance(concept, list):
            # Join list elements into a string or use the first element
            concept = ' '.join(concept) if concept else ''
            
        concept_lower = concept.lower()
        
        if any(indicator in concept_lower for indicator in 
               ['algorithm', 'system', 'framework', 'architecture', 'model']):
            return 'technical_components'
            
        elif any(indicator in concept_lower for indicator in 
                ['method', 'approach', 'procedure', 'protocol']):
            return 'methodological_aspects'
            
        elif any(indicator in concept_lower for indicator in 
                ['experiment', 'test', 'trial', 'measurement']):
            return 'experimental_elements'
            
        return 'theoretical_framework'

    def _parse_non_json_response(self, response: str) -> Dict:
        """
        Parse a non-JSON response into a structured dictionary.
        """
        result = {
            "detailed_description": "",
            "key_components": [],
            "technical_details": [],
            "implications": [],
            "limitations": []
        }
        
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            lower_line = line.lower()
            if 'detailed description' in lower_line:
                current_section = "detailed_description"
            elif 'key components' in lower_line:
                current_section = "key_components"
            elif 'technical details' in lower_line:
                current_section = "technical_details"
            elif 'implications' in lower_line:
                current_section = "implications"
            elif 'limitations' in lower_line:
                current_section = "limitations"
            elif current_section:
                # Remove bullet points and other markers
                clean_line = line.lstrip('•-*> ')
                if clean_line:
                    if current_section == "detailed_description":
                        if result[current_section]:
                            result[current_section] += " " + clean_line
                        else:
                            result[current_section] = clean_line
                    else:
                        result[current_section].append(clean_line)
        
        return result

    def deep_dive_analysis(self, paper_content: str, knowledge_structure: Dict) -> Dict:
        """
        Third analysis stage: Detailed analysis of each major component.
        """
        self.logger.info("Performing deep dive analysis...")
        analyses = {}
        
        # Handle case where concept_hierarchies might be a list of lists or strings
        concepts = knowledge_structure.get('concept_hierarchies', [])
        if not concepts:
            self.logger.warning("No concepts found in knowledge structure")
            return analyses

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Analyzing components...", total=len(concepts))
            
            for concept in concepts:
                # Convert concept to string if it's a list or other type
                concept_str = concept if isinstance(concept, str) else str(concept)
                progress.update(task, advance=1, description=f"Analyzing: {concept_str[:30]}...")
                
                # Determine the type of analysis needed
                analysis_type = self._determine_analysis_type(concept)
                prompt_template = self.config['analysis_framework']['deep_dive'][analysis_type]['instructions']
                
                prompt = f"""
                Perform a detailed analysis of this concept:
                {concept_str}

                {prompt_template}

                Provide your analysis in JSON format with these sections:
                {{
                    "detailed_description": "description here",
                    "key_components": ["component1", "component2"],
                    "technical_details": ["detail1", "detail2"],
                    "implications": ["implication1", "implication2"],
                    "limitations": ["limitation1", "limitation2"]
                }}

                Paper content:
                {paper_content[:4000]}
                """
                
                response = self.query_ollama(prompt, temperature=0.4)
                try:
                    # Try to extract JSON if present
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = response[start_idx:end_idx]
                        analyses[concept_str] = json.loads(json_str)
                    else:
                        raise json.JSONDecodeError("No JSON found", response, 0)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse JSON for concept: {concept_str}")
                    # Create structured output from text
                    analyses[concept_str] = self._parse_non_json_response(response)
                
                time.sleep(1)  # Rate limiting
                
        return analyses

    def synthesize_findings(self, paper_content: str, all_analyses: Dict) -> str:
        """
        Final analysis stage: Synthesize all findings into a cohesive analysis.
        """
        self.logger.info("Synthesizing findings...")
        
        analyses_summary = json.dumps(all_analyses, indent=2)
        prompt = f"""
        {self.config['analysis_framework']['synthesis']['instructions']}

        Based on these analyses:
        {analyses_summary}

        Generate a comprehensive markdown-formatted analysis following this structure:
        {self.config['output_format']['structure']}

        Ensure the analysis is:
        1. Detailed yet clear
        2. Well-structured with appropriate headers
        3. Includes specific examples and references from the paper
        4. Highlights key innovations and implications

        Paper content (for reference):
        {paper_content[:4000]}
        """
        
        return self.query_ollama(prompt, temperature=0.5)

    def analyze_paper(self, paper_path: str) -> Optional[str]:
        """
        Main method to analyze a paper through all stages.
        
        Args:
            paper_path: Path to the PDF file to analyze
            
        Returns:
            String containing the final analysis or None if there's an error
        """
        try:
            # Read the PDF file
            import PyPDF2
            
            self.logger.info(f"Reading paper: {paper_path}")
            with open(paper_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                paper_content = ""
                for page in reader.pages:
                    paper_content += page.extract_text()
                    
            if not paper_content.strip():
                self.logger.error("Failed to extract text from PDF")
                return None
                
            # Perform analysis stages
            domain_analysis = self.analyze_domain(paper_content)
            knowledge_structure = self.map_knowledge_structure(paper_content, domain_analysis)
            detailed_analysis = self.deep_dive_analysis(paper_content, knowledge_structure)
            
            final_analysis = self.synthesize_findings(paper_content, {
                "domain": domain_analysis,
                "structure": knowledge_structure,
                "details": detailed_analysis
            })
            
            return final_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing paper: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Adaptive Paper Analyzer")
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    parser.add_argument("--config", default="adaptive_paper_analyzer.yaml",
                       help="Path to YAML configuration file")
    parser.add_argument("--model", default="llama3.1:8b",
                       help="Name of the Ollama model to use")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AdaptivePaperAnalyzer(args.config, args.model)
    
    # Analyze paper
    analysis = analyzer.analyze_paper(args.pdf_path)
    
    if analysis:
        if args.output:
            # Save to file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(analysis)
            print(f"\nAnalysis saved to: {output_path}")
        else:
            # Print to console
            print("\nPaper Analysis:")
            print(analysis)
    else:
        print("\nFailed to analyze paper")
        sys.exit(1)

if __name__ == "__main__":
    main()