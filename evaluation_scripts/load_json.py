import json

def extract_function_body(json_data, function_name):
    """
    Extract the function body from a JSON dictionary given the function name.
    """
    if json_data.get("function_name") == function_name:
        return json_data.get("function_body", "")
    else:
        raise ValueError(f"Function '{function_name}' not found in JSON.")

def main():
    input_file = "input.json"
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    function_name = "craft"
    body = extract_function_body(data, function_name)

    print(f"\nFunction '{function_name}' body:\n")
    print(body)

if __name__ == "__main__":
    main()
