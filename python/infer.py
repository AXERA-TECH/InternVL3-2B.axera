from llm import LLM
import argparse


if __name__ == "__main__":

    prompt = None
    parser = argparse.ArgumentParser(description="Model configuration parameters")
    parser.add_argument("--hf_model", type=str, default="./InternVL3-2B",
                        help="Path to HuggingFace model")
    parser.add_argument("--axmodel_path", type=str, default="./InternVL3-2B_axmodel",
                        help="Path to save compiled axmodel of llama model")
    parser.add_argument("--vit_model", type=str, default=None,
                        help="Path to save compiled axmodel of llama model")
    parser.add_argument("--sources", nargs='+', type=str, default=None,
                        help="Path to the test image.")
    parser.add_argument("-q", "--question", type=str, default="Please calculate the derivative of the function y=2x^2.",
                        help="Your question that you want to ask the model.")
    args = parser.parse_args()


    hf_model_path = args.hf_model
    axmodel_path = args.axmodel_path
    vit_axmodel_path = args.vit_model
    test_imgs_path = args.sources
    
    llm = LLM(hf_model_path, axmodel_path, vit_axmodel_path)

    for msg in llm.generate(test_imgs_path, args.question):
        print(msg, end="", flush=True)
    
    print("\n\n\n")