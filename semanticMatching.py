import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import torch

class MatchingFunction:
    def __init__(self):
        self.dfSets = {}
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    '''
    datasetA - str: filename of the dataset
    datasetB - str: filename of the dataset
    descriptionsListA - str: description
    descriptionsListB - str: description
    filteredRowsA - dictionary(item - string, values - list of string): item is columns, values is the value of the columns
    filteredRowsB - dictionary(item - string, values - list of string): item is columns, values is the value of the columns
    rowsToPrintA - list of str: list of name to include in the tables aside from definition
    rowsToPrintB - list of str: list of name to include in the tables aside from definition
    sheetNameA - str/int: sheetname for the file
    sheetNameB - str/int: sheetname for the file
    headerA - int: header of the file
    headerB - int: header of the file
    '''
    def cosineSimilarity(self, datasetA, datasetB, descriptionA, descriptionB, filteredRowsA = {}, 
                         filteredRowsB = {}, rowsToPrintA = [], rowsToPrintB = [], 
                         sheetNameA = 0, sheetNameB = 0, headerA = 0, headerB = 0, variableNameA = [], variableNameB = []):
        if datasetA not in self.dfSets:
            self.dfSets[datasetA] = pd.read_excel(datasetA, sheet_name=sheetNameA, header=headerA)
        if datasetB not in self.dfSets:
            self.dfSets[datasetB] = pd.read_excel(datasetB, sheet_name=sheetNameB, header=headerB)
        
        dfA = self.dfSets[datasetA][[descriptionA] + rowsToPrintA + list(filteredRowsA.keys()) + variableNameA]
        dfB = self.dfSets[datasetB][[descriptionB] + rowsToPrintB + list(filteredRowsB.keys()) + variableNameB]
        
        if descriptionA not in dfA.columns:
            print('no description for datasetA')
            return None
        if descriptionB not in dfB.columns:
            print('no description for datasetB')
            return None
        print(f'Filtering the rows')
        # Filtering rows based on conditions
        for item, values in filteredRowsA.items():
            if item in dfA.columns:
                dfA = dfA[dfA[item].isin(values)]
        for item, values in filteredRowsB.items():
            if item in dfB.columns:
                dfB = dfB[dfB[item].isin(values)]

        print(f"Shape of dfA after filtering: {dfA.shape}")
        print(f"Shape of dfB after filtering: {dfB.shape}")
        
        #dfA.loc[:, descriptionA] = dfA[descriptionA].fillna('')
        #dfB.loc[:, descriptionB] = dfB[descriptionB].fillna('')

        dfA = dfA.dropna(subset=[descriptionA]).reset_index(drop=True)
        dfB = dfB.dropna(subset=[descriptionB]).reset_index(drop=True)

        print(f"Shape of dfA after dropna: {dfA.shape}")
        print(f"Shape of dfB after dropna: {dfB.shape}")
        
        print(f'model learning ...')
        # Encode the existing descriptions from the MDR dataset into embeddings
        definitionEmbeddingsB = self.model.encode(dfB[descriptionB].tolist())
        definitionEmbeddingsA = self.model.encode(dfA[descriptionA].tolist())
        # Initialize a list to store results
        results = []

        for i, embeddingA in enumerate(definitionEmbeddingsA):
            similarities = util.pytorch_cos_sim(embeddingA, definitionEmbeddingsB)
            most_similar_idx = similarities.argmax().item()
            similarity_score = round(similarities[0][most_similar_idx].item() * 100, 1)
            result = {
                "VariableNameA": dfA.iloc[i][variableNameA[0]] if variableNameA and variableNameA[0] in dfA.columns else None,
                "VariableNameB": dfB.iloc[most_similar_idx][variableNameB[0]] if variableNameB and variableNameB[0] in dfB.columns else None,
                "descriptionA": dfA[descriptionA].iloc[i],
                "descriptionB": dfB[descriptionB].iloc[most_similar_idx],
                "similarity_score": similarity_score
            }
            if rowsToPrintA:
                for col in rowsToPrintA:
                    result[f"{datasetA}: {col}"] = dfA.iloc[i][col]

            if rowsToPrintB:
                for col in rowsToPrintB:
                    result[f"{datasetB}: {col}"] = dfB.iloc[most_similar_idx][col]

            results.append(result)

        results_df = pd.DataFrame(results)
        sorted_df = results_df.sort_values(by=['similarity_score'], ascending=[False])
        sorted_df['similarity_score'] = sorted_df['similarity_score'].apply(lambda x: str(x) + '%')
        sorted_df = sorted_df.drop_duplicates(subset=["descriptionA", "descriptionB"]).reset_index(drop=True)
        return sorted_df
    '''
    datasetA - str: filename of the dataset
    datasetB - str: filename of the dataset
    descriptionsListA - list of string: list of columns to take in as description
    descriptionsListB - list of string: list of columns to take in as description
    filteredRowsA - dictionary(item - string, values - list of string): item is columns, values is the value of the columns
    filteredRowsB - dictionary(item - string, values - list of string): item is columns, values is the value of the columns
    sheetNameA - str/int: sheetname for the file
    sheetNameB - str/int: sheetname for the file
    headerA - int: header of the file
    headerB - int: header of the file
    '''
    def cosineSimilarityMultiple(self, datasetA, datasetB, descriptionsListA, descriptionsListB, 
                                 filteredRowsA={}, filteredRowsB={}, rowsToPrintA=[], rowsToPrintB=[], 
                                 sheetNameA=0, sheetNameB=0, headerA=0, headerB=0
                                 , variableNameA = [""], variableNameB = [""]):
        print(f'Loading the data')
        if datasetA not in self.dfSets:
            self.dfSets[datasetA] = pd.read_excel(datasetA, sheet_name=sheetNameA, header=headerA)
        if datasetB not in self.dfSets:
            self.dfSets[datasetB] = pd.read_excel(datasetB, sheet_name=sheetNameB, header=headerB)
        
        dfA = self.dfSets[datasetA][descriptionsListA + rowsToPrintA + list(filteredRowsA.keys()) + variableNameA]
        dfB = self.dfSets[datasetB][descriptionsListB + rowsToPrintB + list(filteredRowsB.keys()) + variableNameB]

        print(f"Shape of dfA before filtering: {dfA.shape}")
        print(f"Shape of dfB before filtering: {dfB.shape}")
        
        print(f'Filtering out rows')
        # Apply filtering
        for item, values in filteredRowsA.items():
            if item in dfA.columns:
                dfA = dfA[dfA[item].isin(values)]
        for item, values in filteredRowsB.items():
            if item in dfB.columns:
                dfB = dfB[dfB[item].isin(values)]

        print(f"Shape of dfA after filtering: {dfA.shape}")
        print(f"Shape of dfB after filtering: {dfB.shape}")

        print(f'Model is evaluating')

        # Combine description columns into a single column
        dfA['totalDescription'] = dfA[descriptionsListA].astype(str).apply(lambda row: ' '.join(row.values).strip(), axis=1)
        dfB['totalDescription'] = dfB[descriptionsListB].astype(str).apply(lambda row: ' '.join(row.values).strip(), axis=1)

        # Explicitly remove empty descriptions to prevent skipped rows
        dfA = dfA[dfA['totalDescription'] != 'nan'].reset_index(drop=True)
        dfB = dfB[dfB['totalDescription'] != 'nan'].reset_index(drop=True)

        print(f"Shape of dfA after removing empty descriptions: {dfA.shape}")
        print(f"Shape of dfB after removing empty descriptions: {dfB.shape}")

        # Compute embeddings
        definitionEmbeddingsA = self.model.encode(dfA['totalDescription'].tolist(), convert_to_tensor=True)
        definitionEmbeddingsB = self.model.encode(dfB['totalDescription'].tolist(), convert_to_tensor=True)

        results = []

        # Compute cosine similarities using a for-loop
        for i, embeddingA in enumerate(definitionEmbeddingsA):
            similarities = util.pytorch_cos_sim(embeddingA, definitionEmbeddingsB)  # Compute similarity for one row
            most_similar_idx = similarities.argmax().item()  # Get the best match index
            similarity_score = round(similarities[0][most_similar_idx].item() * 100, 1)

            result = {
                "VariableNameA": dfA.iloc[i][variableNameA[0]] if variableNameA and variableNameA[0] in dfA.columns else None,
                "VariableNameB": dfB.iloc[most_similar_idx][variableNameB[0]] if variableNameB and variableNameB[0] in dfB.columns else None,
                "descriptionA": dfA['totalDescription'].iloc[i],
                "descriptionB": dfB['totalDescription'].iloc[most_similar_idx],
                "similarity_score": similarity_score
            }

            # Add additional metadata if requested
            for col in rowsToPrintA:
                if col in dfA.columns:
                    result[f"{datasetA}: {col}"] = dfA.iloc[i][col]

            for col in rowsToPrintB:
                if col in dfB.columns:
                    result[f"{datasetB}: {col}"] = dfB.iloc[most_similar_idx][col]

            results.append(result)

        results_df = pd.DataFrame(results)

        # Sort results by similarity score
        sorted_df = results_df.sort_values(by=['similarity_score'], ascending=[False])
        sorted_df['similarity_score'] = sorted_df['similarity_score'].apply(lambda x: str(x) + '%')
        sorted_df = sorted_df.drop_duplicates(subset=["descriptionA", "descriptionB"]).reset_index(drop=True)
        return sorted_df

    def fast_jaccard_similarity(self, X1, X2):
        intersection = X2 @ X1.T  # Fast sparse matrix multiplication
        union = X2.sum(axis=1)[:, None] + X1.sum(axis=1)[None, :] - intersection
        return intersection / union  # Keep it as a sparse matrix
    '''
    datasetA - str: filename of the dataset
    datasetB - str: filename of the dataset
    descriptionsListA - str: description
    descriptionsListB - str: description
    filteredRowsA - dictionary(item - string, values - list of string): item is columns, values is the value of the columns
    filteredRowsB - dictionary(item - string, values - list of string): item is columns, values is the value of the columns
    rowsToPrintA - list of str: list of name to include in the tables aside from definition
    rowsToPrintB - list of str: list of name to include in the tables aside from definition
    sheetNameA - str/int: sheetname for the file
    sheetNameB - str/int: sheetname for the file
    headerA - int: header of the file
    headerB - int: header of the file
    '''
    def cosineJaccardSimilarity(self, datasetA, datasetB, descriptionA, descriptionB, 
                                filteredRowsA = {}, filteredRowsB = {}, rowsToPrintA=[], 
                                rowsToPrintB=[], sheetNameA=0, sheetNameB=0, headerA=0, headerB=0, 
                                variableNameA = [""], variableNameB = [""]):
        print(f'Loading the data')
        if datasetA not in self.dfSets:
            self.dfSets[datasetA] = pd.read_excel(datasetA, sheet_name=sheetNameA, header=headerA)
        if datasetB not in self.dfSets:
            self.dfSets[datasetB] = pd.read_excel(datasetB, sheet_name=sheetNameB, header=headerB)

        dfA = self.dfSets[datasetA][variableNameA + [descriptionA] + rowsToPrintA + list(filteredRowsA.keys())]
        dfB = self.dfSets[datasetB][variableNameB + [descriptionB] + rowsToPrintB + list(filteredRowsB.keys())]

        if descriptionA not in dfA.columns:
            print('No description for datasetA')
            return None
        if descriptionB not in dfB.columns:
            print('No description for datasetB')
            return None
        
        print(f'Filtering the rows')

        for item, values in filteredRowsA.items():
            if item in dfA.columns:
                dfA = dfA[dfA[item].isin(values)]
        for item, values in filteredRowsB.items():
            if item in dfB.columns:
                dfB = dfB[dfB[item].isin(values)]

        print(f"Shape of dfA after filtering: {dfA.shape}")
        print(f"Shape of dfB after filtering: {dfB.shape}")

        dfA = dfA.dropna(subset=[descriptionA]).reset_index(drop=True)
        dfB = dfB.dropna(subset=[descriptionB]).reset_index(drop=True)

        print(f"Shape of dfA after dropna: {dfA.shape}")
        print(f"Shape of dfB after dropna: {dfB.shape}")

        print(f'Model learning ...')

        dfAEmbeddings = self.model.encode(dfA[descriptionA].tolist(), convert_to_tensor=True)
        dfBEmbeddings = self.model.encode(dfB[descriptionB].tolist(), convert_to_tensor=True)

        # Compute cosine similarity
        similarity_matrix = util.cos_sim(dfAEmbeddings, dfBEmbeddings)

        num_matches = 1  # Only find 1 match per description
        top_n_match_indices = torch.argsort(similarity_matrix, dim=1, descending=True)[:, :num_matches]

        vectorizer = CountVectorizer(binary=True, stop_words="english")
        dfAsparse = vectorizer.fit_transform(dfA[descriptionA])
        dfBsparse = vectorizer.transform(dfB[descriptionB])

        cosine_weight = 0.85
        jaccard_weight = 0.15

        expanded_rows = []

        for row_idx in range(dfAsparse.shape[0]):
            row = dfA.iloc[row_idx]

            match_idx = top_n_match_indices[row_idx, 0].item()

            combined_score = 0.0
            cosine_score = 0.0
            jaccard_score_value = 0.0
            matched_mdr_definition = None
            matched_row_data = []

            if match_idx < dfB.shape[0]:
                matched_row = dfB.iloc[match_idx]
                matched_mdr_definition = matched_row[descriptionB]
                cosine_score = similarity_matrix[row_idx, match_idx].item()
                
                jaccard_score_value = self.fast_jaccard_similarity(dfAsparse[row_idx], dfBsparse[match_idx]).toarray()[0, 0]

                combined_score = (cosine_score * cosine_weight) + (jaccard_score_value * jaccard_weight)

                # Collect additional row info from datasetB
                matched_row_data = [matched_row[col] for col in rowsToPrintB]

            # Combine all data into a single row
            full_row = [row[col] for col in variableNameA] + [matched_row[col] for col in variableNameB] + [
                row[descriptionA],  # Original description
                matched_mdr_definition,  # Matched description
                round(combined_score * 100, 2),  # Combined score
                round(cosine_score * 100, 2),  # Cosine similarity score
                round(jaccard_score_value * 100, 2)  # Jaccard similarity score
            ]
            full_row.extend([row[col] for col in rowsToPrintA])
            full_row.extend(matched_row_data if matched_row_data else [""] * len(rowsToPrintB))

            expanded_rows.append(full_row)

        # Debugging prints to check column alignment
        expected_columns = ["VariableNameA", "VariableNameB", "descriptionA", "descriptionB", "similarity_score", "Cosine Score", "Jaccard Score"] + rowsToPrintA + rowsToPrintB
        print(f"Expected columns count: {len(expected_columns)}")
        print(f"Actual row length: {len(expanded_rows[0]) if expanded_rows else 'No data'}")
        print(expanded_rows[0])

        df_final = pd.DataFrame(expanded_rows, columns=expected_columns)

        print("Processing completed.")
        
        df_final = df_final.sort_values(by=["similarity_score"], ascending=False).reset_index(drop=True)
        df_final['similarity_score'] = df_final['similarity_score'].apply(lambda x: str(x) + '%')
        df_final['Cosine Score'] = df_final['Cosine Score'].apply(lambda x: str(x) + '%')
        df_final['Jaccard Score'] = df_final['Jaccard Score'].apply(lambda x: str(x) + '%')
        df_final = df_final.drop_duplicates(subset=["descriptionA", "descriptionB"]).reset_index(drop=True)
        return df_final



    def compare_csv_files(self, csv_files, output_csv="comparison_results.csv"):
        """
        Compare multiple CSV files and display differences side by side, matching only on VariableNameA while maintaining row order.

        :param csv_files: Dictionary where keys are CSV filenames and values are lists containing score column names
                        Example: {"file1.csv": ["VariableNameB", "Similarity Score", "Actual Result"], "file2.csv": ["VariableNameB", "Similarity Score", "Actual Result"]}
        :param output_csv: Name of the output CSV file (default: "comparison_results.csv")
        :return: None
        """
        dataframes = []
        file_list = list(csv_files.keys())  # Convert dict_keys to a list

        for file, columns in csv_files.items():
            # Read CSV while preserving row order
            df = pd.read_csv(file)

            # Keep only relevant columns
            df = df[["VariableNameA"] + columns]

            # Rename columns to include file name
            df.rename(columns={
                columns[0]: f"VariableNameB ({file})",
                columns[1]: f"Similarity Score ({file})",
                columns[2]: f"Actual Result ({file})"
            }, inplace=True)

            dataframes.append(df)

        # Start merging using the first dataframe as the base
        merged_df = dataframes[0]
        expected_rows = len(merged_df)  # Expected row count from the first file
        print(f"Initial rows: {expected_rows}")

        for i, df in enumerate(dataframes[1:], start=2):
            print(f"Merging file {file_list[i-1]}: {len(df)} rows")
            merged_df = pd.merge(merged_df, df, on=["VariableNameA"], how="outer", sort=False)
            print(f"After merging file {file_list[i-1]}, total rows: {len(merged_df)}")

        # Ensure only `VariableNameA` values that exist in all files are retained
        for file in file_list:
            merged_df = merged_df[merged_df[f"VariableNameB ({file})"].notna()]  # Ensure all VariableNameB exist

        # Drop duplicate rows while keeping the first occurrence
        merged_df = merged_df.drop_duplicates(subset=["VariableNameA"]).reset_index(drop=True)

        # Restore original sorting order based on the first file
        merged_df = merged_df.set_index("VariableNameA").reindex(dataframes[0]["VariableNameA"]).reset_index()

        # Create match columns
        for file in file_list:
            merged_df[f"Match ({file})"] = (merged_df["VariableNameA"] == merged_df[f"Actual Result ({file})"]).astype(int)

        # Save the final comparison results
        merged_df.to_csv(output_csv, index=False)
        print(f"Comparison CSV file created: {output_csv}")



if __name__ == '__main__':
    matcher = MatchingFunction()

    # results = matcher.cosineSimilarityMultiple(
    #     datasetB="data/ABS-MOPS Variables - December 11 2024.xlsm",
    #     datasetA="data/MDR View 02-27-2025.xlsx",
    #     descriptionsListB=['Provide a brief description of the variable, this will alert staff entering content of its intended purpose\n\nNOTE: Maximum number of characters should be 500'],
    #     descriptionsListA=["description"],
    #     rowsToPrintB=['Unique Name for Variable \nOn upload, will verify with those already in database to ensure unique and alert to those that are not\n\nNOTE: \n1) Variable Names should be all caps with no spaces\n2) Variables can not end with _# or _## as those are reserved for handling of repeating persons.'],
    #     variableNameA=["variable_name"],
    #     variableNameB=["Legacy Variable"],
    #     sheetNameB='Data Sheet',
    #     headerB=13,
    #     filteredRowsA={"frame": ["Business Frame"]}
    # )
    # results.to_csv('result/cosineSimilarityMultiple.csv')
    # results = matcher.cosineSimilarity(
    #     datasetB="data/ABS-MOPS Variables - December 11 2024.xlsm",
    #     datasetA="data/MDR View 02-27-2025.xlsx",
    #     descriptionB='Provide a brief description of the variable, this will alert staff entering content of its intended purpose\n\nNOTE: Maximum number of characters should be 500',
    #     descriptionA="description",
    #     rowsToPrintB=['Unique Name for Variable \nOn upload, will verify with those already in database to ensure unique and alert to those that are not\n\nNOTE: \n1) Variable Names should be all caps with no spaces\n2) Variables can not end with _# or _## as those are reserved for handling of repeating persons.'],
    #     variableNameA=["variable_name"],
    #     variableNameB=["Legacy Variable"],
    #     sheetNameB='Data Sheet',
    #     headerB=13,
    #     filteredRowsA={"frame": ["Business Frame"]}
    # )
    
    # results.to_csv('result/cosineSimilarity.csv')
    
    # results = matcher.cosineJaccardSimilarity(
    #     datasetB="data/ABS-MOPS Variables - December 11 2024.xlsm",
    #     datasetA="data/MDR View 02-27-2025.xlsx",
    #     descriptionB='Provide a brief description of the variable, this will alert staff entering content of its intended purpose\n\nNOTE: Maximum number of characters should be 500',
    #     descriptionA="description",
    #     rowsToPrintB=['Unique Name for Variable \nOn upload, will verify with those already in database to ensure unique and alert to those that are not\n\nNOTE: \n1) Variable Names should be all caps with no spaces\n2) Variables can not end with _# or _## as those are reserved for handling of repeating persons.'],
    #     variableNameA=["variable_name"],
    #     variableNameB=["Legacy Variable"],
    #     sheetNameB='Data Sheet',
    #     headerB=13,
    #     filteredRowsA={"frame": ["Business Frame"]}
    # )
    
    # results.to_csv('result/cosineJaccardSimilarity.csv')

    
    
    csv_files = {
        "result/cosineSimilarity.csv": ["VariableNameB","similarity_score", "result"],
        "result/cosineJaccardSimilarity.csv": ["VariableNameB","similarity_score", "result"],
        "result/cosineSimilarityMultiple.csv": ["VariableNameB","similarity_score", "result"],
    }

    matcher.compare_csv_files(csv_files, output_csv="result/comparison_results.csv")