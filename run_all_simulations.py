"""
Master Script to Run All ResNet and Big Data Computer Vision Simulations
This script orchestrates the execution of all simulations and generates a comprehensive report.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import warnings

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def print_header(title):
    """Print a formatted header"""
    print("\\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def print_section(title):
    """Print a formatted section header"""
    print("\\n" + "-" * 60)
    print(f" {title}")
    print("-" * 60)

def run_simulation(script_path, simulation_name):
    """
    Run a single simulation script
    
    Args:
        script_path: Path to the simulation script
        simulation_name: Name of the simulation for reporting
        
    Returns:
        Boolean indicating success
    """
    print_section(f"Running {simulation_name}")
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        # Change to the simulations directory
        original_dir = os.getcwd()
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        if script_dir:
            os.chdir(script_dir)
        
        # Run the simulation
        print(f"🚀 Starting {simulation_name}...")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        # Restore original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"✅ {simulation_name} completed successfully!")
            return True
        else:
            print(f"❌ {simulation_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {simulation_name} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {simulation_name}: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print_section("Checking Dependencies")
    
    required_modules = [
        'tensorflow',
        'numpy', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'PIL',
        'cv2'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - Missing")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\\n⚠️  Missing dependencies: {', '.join(missing_modules)}")
        print("Please run: python setup_environment.py")
        return False
    
    print("\\n✅ All dependencies are available!")
    return True

def create_experiment_summary():
    """Create a summary of the experiment results"""
    print_section("Creating Experiment Summary")
    
    summary_content = f"""# ResNet and Big Data Computer Vision - Experiment Summary
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Simulations Completed

### Simulation 4: Residual Blocks vs Standard Blocks
**Objective**: Demonstrate the effectiveness of skip connections in deep networks

**Key Findings**:
- Residual blocks help mitigate vanishing gradient problems
- Skip connections provide direct gradient paths to earlier layers
- Residual networks train more smoothly and achieve better convergence
- Identity mapping allows networks to learn more complex representations

**Generated Files**:
- Training comparison plots
- Gradient magnitude analysis
- Model architecture diagrams

### Simulation 5: Transfer Learning with Pre-trained ResNet
**Objective**: Show how big data (ImageNet) can benefit small dataset problems

**Key Findings**:
- Feature extraction using frozen pre-trained weights is highly effective
- Fine-tuning can further improve performance when done carefully
- Transfer learning significantly outperforms training from scratch
- Big data pre-training provides valuable general feature extractors

**Generated Files**:
- Model performance comparisons
- Training efficiency analysis
- Parameter usage statistics

## File Locations
- **Plots**: `results/` directory
- **Models**: Saved in each simulation's execution
- **Logs**: Training logs and metrics

## Key Concepts Demonstrated
1. **Skip Connections**: Core innovation of ResNet architecture
2. **Vanishing Gradients**: How residual blocks solve this problem
3. **Transfer Learning**: Leveraging big data for small problems
4. **Feature Extraction vs Fine-tuning**: Different transfer learning strategies
5. **Big Data Impact**: How large-scale pre-training benefits all tasks

## Analysis Questions for Reports
1. Why do residual networks train more effectively than standard deep networks?
2. How do skip connections affect gradient flow in deep networks?
3. What advantages does transfer learning provide over training from scratch?
4. When should you use feature extraction vs fine-tuning?
5. How does big data pre-training benefit computer vision tasks?

## Next Steps
1. Analyze the generated plots and results
2. Write detailed reports for each simulation
3. Consider extending experiments with real datasets
4. Explore other pre-trained architectures (VGG, InceptionV3, etc.)

---
*This summary was automatically generated by the ResNet simulation suite.*
"""
    
    # Save summary
    summary_path = "results/experiment_summary.md"
    os.makedirs("results", exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"📝 Experiment summary saved to: {summary_path}")

def main():
    """Main function to orchestrate all simulations"""
    parser = argparse.ArgumentParser(description='Run ResNet and Big Data Computer Vision Simulations')
    parser.add_argument('--sim4-only', action='store_true', help='Run only Simulation 4')
    parser.add_argument('--sim5-only', action='store_true', help='Run only Simulation 5')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    parser.add_argument('--quick', action='store_true', help='Run simulations with reduced epochs for quick testing')
    
    args = parser.parse_args()
    
    # Print welcome message
    print_header("ResNet and Big Data Computer Vision Simulations")
    print(f"🕐 Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("👥 Support Groups: 7, 8, 9")
    print("📚 Course: BCS-05-0839/2022")
    
    # Check dependencies unless skipped
    if not args.skip_deps:
        if not check_dependencies():
            print("\\n❌ Cannot proceed without required dependencies.")
            print("Please run: python setup_environment.py")
            return False
    
    # Define simulations to run
    simulations = []
    
    if args.sim5_only:
        simulations = [
            ("simulations/simulation5_transfer_learning.py", "Simulation 5: Transfer Learning")
        ]
    elif args.sim4_only:
        simulations = [
            ("simulations/simulation4_residual_blocks.py", "Simulation 4: Residual Blocks")
        ]
    else:
        simulations = [
            ("simulations/simulation4_residual_blocks.py", "Simulation 4: Residual Blocks"),
            ("simulations/simulation5_transfer_learning.py", "Simulation 5: Transfer Learning")
        ]
    
    # Set environment variable for quick mode
    if args.quick:
        os.environ['QUICK_MODE'] = '1'
        print("⚡ Quick mode enabled - simulations will use reduced epochs")
    
    # Run simulations
    print_section("Simulation Execution Plan")
    for script_path, name in simulations:
        print(f"📋 {name}")
    
    print(f"\\n🎯 Total simulations to run: {len(simulations)}")
    
    # Confirm execution
    if not args.quick:
        response = input("\\n🚀 Ready to start simulations? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("❌ Execution cancelled by user")
            return False
    
    # Execute simulations
    start_time = datetime.now()
    successful_runs = 0
    
    for script_path, simulation_name in simulations:
        if run_simulation(script_path, simulation_name):
            successful_runs += 1
        else:
            print(f"⚠️  {simulation_name} encountered issues")
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    # Generate summary
    create_experiment_summary()
    
    # Print final results
    print_header("EXECUTION SUMMARY")
    print(f"⏱️  Total execution time: {execution_time}")
    print(f"✅ Successful simulations: {successful_runs}/{len(simulations)}")
    print(f"📁 Results saved in: results/ directory")
    
    if successful_runs == len(simulations):
        print("\\n🎉 All simulations completed successfully!")
        print("\\n📋 Next steps for your assignment:")
        print("1. 📊 Review the generated plots in the results/ directory")
        print("2. 📝 Read the experiment_summary.md file")
        print("3. 🔍 Analyze the training curves and performance metrics")
        print("4. ✍️  Write your detailed analysis and discussion")
        print("5. 🤔 Answer the key questions about ResNet and transfer learning")
        
        print("\\n💡 Key concepts to discuss in your report:")
        print("   • How skip connections solve vanishing gradients")
        print("   • Why residual networks train more effectively")
        print("   • The power of transfer learning and big data")
        print("   • When to use feature extraction vs fine-tuning")
        
        return True
    else:
        print(f"\\n⚠️  {len(simulations) - successful_runs} simulation(s) failed")
        print("Check the error messages above and ensure all dependencies are installed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\n❌ Unexpected error: {e}")
        sys.exit(1)