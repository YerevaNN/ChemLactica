import re


def check_valid_start(input_string, valid_options):
    pattern = r"^\[(?:" + "|".join(valid_options) + r")\s"
    if re.match(pattern, input_string):
        return True
    elif input_string == """[START_SMILES] """:
        return True
    else:
        return False


def check_prompt(input_string, valid_options):
    if not input_string.startswith("["):
        raise ValueError(
            "Prompt must start with '[' character, check if there is a space"
        )
    result = re.split(r"(?<=\])(?=\[)", input_string)
    pattern = r"\[(" + "|".join(valid_options) + r") [^\]]+\]"
    counter = 0
    last_passed = False
    for index, substring in enumerate(result):
        if index == len(result) - 1:
            if " ]" in substring:
                raise ValueError(
                    f"There should not be a space before the ']' character in {substring}"
                )
            if not check_valid_start(substring, valid_options):
                raise ValueError(
                    f"""{substring} as the last property in the prompt,
                    should have a space before asking the model to
                    generate the continuation or is not a valid property"""
                )
            else:
                counter += 1
                last_passed = True
        elif " ]" in substring or "] " in substring:
            raise ValueError(
                f"There should not be a space before or after the ']' character in {substring}"
            )
        elif not re.match(pattern, substring):
            raise ValueError(
                f"{substring} is not a valid property or no value was provided"
            )
        else:
            counter += 1
    if not last_passed:
        raise ValueError(
            f"""The end of the prompt either doesn't end with a space
            or is using an invalid property in {substring}"""
        )

    if counter == len(result):
        return True
    else:
        raise ValueError(
            "There is some formatting issue, likely a space between properties"
        )


# Example usage:
if __name__ == "__main__":
    valid_options = ["CLOGP", "SAS", "WEIGHT", "NUMHACCEPTORS"]
    input_string = "[CLOGP 12.4][WEIGHT dak.3240a][NUMHACCEPTORS "
    result = check_prompt(input_string, valid_options)
