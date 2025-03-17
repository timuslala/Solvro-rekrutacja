import pandas as pd


def split_ingredients(df, max_ingredients=6):
    """
    Splits the 'ingredients' column in the DataFrame into separate columns for each ingredient's details.

    Parameters:
    df (pd.DataFrame): DataFrame containing a column 'ingredients' with a list of ingredient dictionaries.
    max_ingredients (int): Maximum number of ingredients to split into separate columns. Default is 6.
    Returns:
    None. The DataFrame is modified in place.

    The function performs the following steps:
    1. Initializes six new columns (ingredient1 to ingredient6) to None.
    2. If the maximum number of ingredients is 1, the 'ingredients' column is copied to 'ingredient1'.
    3. Otherwise, the function iterates over each row in the DataFrame and splits the 'ingredients' list into separate columns.
    4. Creates new columns for each ingredient's id, name, description, alcohol content, type and measure.
    """
    for i in range(1, max_ingredients + 1):
        df[f"ingredient{i}"] = None
    if max_ingredients == 1:
        df["ingredient1"] = df["ingredients"].copy()
    else:
        for idx, row in df.iterrows():
            ingredients = row.get("ingredients", None)
            if ingredients is not None:
                for i, ingredient in enumerate(ingredients[:max_ingredients], 1):
                    df.at[idx, f"ingredient{i}"] = ingredient
    for i in range(1, max_ingredients + 1):
        df[f"ingredient{i}_id"] = df[f"ingredient{i}"].apply(
            lambda x: x["id"] if x is not None else None
        )
        df[f"ingredient{i}_name"] = df[f"ingredient{i}"].apply(
            lambda x: x["name"] if x is not None else None
        )
        df[f"ingredient{i}_description"] = df[f"ingredient{i}"].apply(
            lambda x: x["description"] if x is not None else None
        )
        df[f"ingredient{i}_alcohol"] = df[f"ingredient{i}"].apply(
            lambda x: x["alcohol"] if x is not None else None
        )
        df[f"ingredient{i}_type"] = df[f"ingredient{i}"].apply(
            lambda x: x["type"] if x is not None else None
        )
        df[f"ingredient{i}_measure"] = df[f"ingredient{i}"].apply(
            lambda x: x.get("measure") if x is not None else None
        )


def replace_ingredient_type(df, ingredient_type, new_ingredient_type):
    """
    Overwrites the ['ingredients'][::]['type'] columns in the DataFrame with a new value.

    Parameters:
    df (pd.DataFrame): DataFrame containing column 'ingredients'.
    ingredient_type (str): The value to be replaced in the ['ingredients'][::]['type'] columns.
    new_ingredient_type (str): New value for the ['ingredients'][::]['type'] columns.
    Returns:
    None. The DataFrame is modified in place.
    """

    def replace_val(row):
        for ingredient in row:
            if (
                isinstance(ingredient, dict)
                and "type" in ingredient
                and ingredient["type"] == ingredient_type
            ):
                ingredient["type"] = new_ingredient_type

    df["ingredients"].apply(replace_val)


def replace_ingredient_type_by_name(df, ingredient_name, new_ingredient_type):
    """
    Overwrites the ['ingredients'][::]['type'] columns in the DataFrame with a new value based on the ingredient name.

    Parameters:
    df (pd.DataFrame): DataFrame containing column 'ingredients'.
    ingredient_name (str): The name of the ingredient to have type replaced.
    new_ingredient_type (str): New value for the ['ingredients'][::]['type'] columns.
    Returns:
    None. The DataFrame is modified in place.
    """

    def replace_val(row):
        for ingredient in row:
            if (
                isinstance(ingredient, dict)
                and "name" in ingredient
                and ingredient["name"] == ingredient_name
            ):
                ingredient["type"] = new_ingredient_type

    df["ingredients"].apply(replace_val)


def drop_ingredient_by_name(df, ingredient_name):
    """
    Drops the ingredient with the specified name from the 'ingredients' column in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing column 'ingredients'.
    ingredient_name (str): The name of the ingredient to be dropped.
    Returns:
    None. The DataFrame is modified in place.
    """

    def drop_ingredient(row):
        return [
            ingredient
            for ingredient in row
            if ingredient.get("name") != ingredient_name
        ]

    df["ingredients"] = df["ingredients"].apply(drop_ingredient)


def drop_ingredient_by_type(df, ingredient_type):
    """
    Drops the ingredient with the specified type from the 'ingredients' column in the DataFrame."

    Parameters:
    df (pd.DataFrame): DataFrame containing column 'ingredients'."
    ingredient_type (str): The type of the ingredient to be dropped.""
    Returns:
    None. The DataFrame is modified in place."
    """

    def drop_ingredient(row):
        return [
            ingredient
            for ingredient in row
            if ingredient.get("type") != ingredient_type
        ]

    df["ingredients"] = df["ingredients"].apply(drop_ingredient)


def replace_measure(df, old_measure, new_measure):
    """
    Replaces the measure value in the ['ingredients'][::]['measure'] columns in the DataFrame with a new value.
    Has an additional option to replace None values with a new measure.
    Parameters:
    df (pd.DataFrame): DataFrame containing column 'ingredients'.
    old_measure (str): The value to be replaced in the ['ingredients'][::]['measure'] columns.
    new_measure (str): New value for the ['ingredients'][::]['measure'] columns.
    Returns:
    None. The DataFrame is modified in place.
    """

    def replace_val(row):
        for ingredient in row:
            if (
                isinstance(ingredient, dict)
                and old_measure is None
                and "measure" not in ingredient
            ):
                ingredient["measure"] = new_measure
            if (
                isinstance(ingredient, dict)
                and "measure" in ingredient
                and ingredient["measure"] == old_measure
            ):
                ingredient["measure"] = new_measure

    df["ingredients"].apply(replace_val)


def perform_weighted_one_hot_encoding(df):
    """
    Performs one-hot encoding on ingredient type columns and weights the encoding by the ingredient measure.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'ingredient1_type' to 'ingredient6_type' and 'ingredient1_measure' to 'ingredient6_measure'.

    Returns:
    None. The DataFrame is modified in place.
    """
    ingredient_types = (
        pd.concat(
            [
                df["ingredient1_type"],
                df["ingredient2_type"],
                df["ingredient3_type"],
                df["ingredient4_type"],
                df["ingredient5_type"],
                df["ingredient6_type"],
            ]
        )
        .dropna()
        .unique()
    )
    for ingredient_type in ingredient_types:
        ingredient_type_name = ingredient_type.lower().replace(" ", "_")
        df[f"ingredient_{ingredient_type_name}"] = 0
        for i in range(1, 7):
            df.loc[
                df[f"ingredient{i}_type"] == ingredient_type,
                f"ingredient_{ingredient_type_name}",
            ] = df[f"ingredient{i}_measure"]
