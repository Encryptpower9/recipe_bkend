require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');

const app = express();
const port = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Client (initialize once)
const client = new MongoClient(process.env.MONGO_URI);
const client2 = new MongoClient(process.env.MONGO_URI2);
let db;
let db2;

// Gemini API setup
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });
const chatModel = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

async function connectToMongo() {
    try {
        await client.connect();
        db = client.db("RecipeDatabase");
        console.log("Connected to dv Atlas!");

        await client2.connect();
        db2 = client2.db("recipes_full_db");
        console.log("Connected to dk Atlas!");

    } catch (error) {
        console.error("MongoDB connection error:", error);
        process.exit(1);
    }
}

// Connect to MongoDB when the server starts
connectToMongo().then(() => {
    app.listen(port, () => {
        console.log(`Server running on port ${port}`);
    });
});

// Simple root route
app.get('/', (req, res) => {
    res.send('Recipe Book Backend is running!');
});

// RAG Search Endpoint
app.post('/api/recipes/search', async (req, res) => {
    const userQuery = req.body.query;
    console.log('Received user query:', userQuery);
    const dietaryRestrictions = req.body.dietaryRestrictions;
    const cuisinePreferences = req.body.cuisinePreferences;
    const mealType = req.body.mealType;

    if (!userQuery || typeof userQuery !== 'string') {
        return res.status(400).json({ error: "Query parameter is required and must be a string." });
    }

    const validFormat = /^[a-zA-Z0-9\s.,?!'"()-]+$/;

    if (!validFormat.test(userQuery) || !/[a-zA-Z]/.test(userQuery.trim())) {
        return res.status(400).json({
            error: "Query must contain only letters, numbers, and spaces, and include at least one Character."
        });
    }

    try {
        // 1. Generate query embedding
        const { embedding } = await embeddingModel.embedContent(userQuery);

        // 2. Perform vector search
        const recipesCollection = db.collection("recipes");
        const aggPipeline = [
            {
                $vectorSearch: {
                    index: process.env.VECTOR_SEARCH_INDEX_NAME,
                    path: "recipe_text_embedding",
                    queryVector: embedding.values,
                    numCandidates: 5000,
                    limit: 5
                }
            },
            {
                $project: {
                    _id: 1,
                    title: 1,
                    ingredients: 1,
                    instructions: 1,
                    score: { "$meta": "vectorSearchScore" }
                }
            }
        ];

        let retrievedRecipes = (await recipesCollection.aggregate(aggPipeline).toArray()).map(recipe => ({
            // This .map call is crucial to ensure _id is a string, even if it's already a string or ObjectId
            ...recipe,
            _id: recipe._id ? recipe._id.toString() : null
        }));

        if (retrievedRecipes.length === 0) {
            return res.status(400).json({ error: "No recipes found for your query." });
        }


        // --- LLM call proceeds ONLY if recipes are found ---
        const chat = chatModel.startChat({
            generationConfig: {
                temperature: 0.1, // Keeping temperature very low for precise formatting
                topK: 1,
                topP: 1,
            },
            safetySettings: [
                {
                    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
            ],
        });

        // Prepare the context for the LLM
        const recipeContextForLLM = retrievedRecipes.map((recipe, index) => {
            const score = recipe.score ? recipe.score.toFixed(4) : 'N/A';
            return `Recipe ${index + 1} - Title: "${recipe.title}", Score: ${score}`;
        }).join('\n');

        // Construct the prompt to instruct the LLM on exact formatting
        const ragPrompt = `You are a helpful recipe assistant. Your task is to list the provided recipes in a very specific format.
        Answer the user's question accurately based *only* on the provided recipe information. If the answer is not contained within the provided context, state that you don't have enough information about recipes; and do not attach any recipes in that case.

        User's original query: "${userQuery}"
        

        Retrieved Recipe Information (Context):
        ${recipeContextForLLM}

        Based *only* on the "Retrieved Recipe Information" provided above, generate a list of these recipes. 
        
        Your response must follow this EXACT format for each recipe:

        Recipe X (Score: Y.YYYY): Recipe Title

        Where X is the recipe number (starting from 1), Y.YYYY is the score to four decimal places, and Recipe Title is the exact title.

        Additionally, remember:

        Ensure the selections adhere to criteria: “${dietaryRestrictions}; ${cuisinePreferences} cuisine; ${mealType}”.
        If the query does not match any recipes, respond with "I don't have enough information about recipes." and do not include any recipes in the response.

        Do NOT include any other text, Just provide the formatted list.`;

        const chatResult = await chat.sendMessage(ragPrompt);
        const llmResponseText = chatResult.response.text();

        

        // 4. Attach/bind image URLs to retrievedRecipes
        const imageCollection = db2.collection("recipe_images_full");

        const validRecipeIds = retrievedRecipes
            .filter(recipe => recipe._id !== null)
            .map(recipe => recipe._id);


        let imageDocs = [];
        if (validRecipeIds.length > 0) {
            // Fetch image documents from the FULL image collection on the new cluster
            imageDocs = await imageCollection.find({ _id: { $in: validRecipeIds } }).toArray();

            /* --- DEBUG POINT 2: Check the documents actually fetched from the image collection ---
            console.log('\n--- DEBUG POINT 2: Documents Fetched from Image Collection ---');
            if (imageDocs.length > 0) {
                // Log _id and image count for each fetched document
                imageDocs.forEach(doc => {
                    console.log(`Fetched Image Doc: _id='${doc._id}', images_count=${doc.images ? doc.images.length : 0}`);
                    if (doc.images && doc.images.length > 0 && doc.images[0].url) {
                        console.log(`  First Image URL: ${doc.images[0].url.substring(0, 70)}...`); // Log truncated URL
                    } else {
                        console.log('  No valid image URL found in this document.');
                    }
                });
            } else {
                console.log('No image documents were fetched for the validRecipeIds.');
            }*/
        } else {
            console.log('No validRecipeIds to search for images (array was empty after filter).');
        }


        // Create a map for quick lookup of image URLs by recipe ID (string)
        const imageUrlMap = new Map();
        imageDocs.forEach(doc => {
            // Ensure doc._id is used as the key, converted to string for consistency
            if (doc.images && doc.images.length > 0 && doc.images[0].url) {
                // --- DEBUG POINT 3: Check what's being mapped into imageUrlMap ---
                //console.log(`DEBUG POINT 3: Mapping _id='${doc._id.toString()}' to URL.`);
                imageUrlMap.set(doc._id.toString(), doc.images[0].url);
            } else {
                //console.log(`DEBUG POINT 3: Skipping mapping for _id='${doc._id.toString()}' due to missing images/url.`);
            }
        });

        /* --- DEBUG POINT 4: Check final size of the imageUrlMap ---
        console.log('\n--- DEBUG POINT 4: Final imageUrlMap Size ---');
        console.log('Final imageUrlMap size:', imageUrlMap.size); */


        // Enrich the original retrievedRecipes array with image URLs.
        let recipesWithImages = retrievedRecipes.map(recipe => {
            let imageUrl = null;

            // Only attempt to get imageUrl if recipe._id is present (and it's guaranteed to be a string or null)
            if (recipe._id) {
                imageUrl = imageUrlMap.get(recipe._id); // Use the already stringified _id for lookup

                // --- DEBUG POINT 5: Check the URL retrieved from the map for each recipe ---
                //console.log(`DEBUG POINT 5: For recipe _id='${recipe._id}', imageUrl from map: ${imageUrl ? imageUrl.substring(0, 70) + '...' : 'null'}`);
            } else {
                //console.log('DEBUG POINT 5: Recipe without _id (imageUrl will be null).');
            }

            return {
                ...recipe,
                imageUrl: imageUrl || null // Assign fetched URL or null if not found
            };
        });

        /* --- DEBUG POINT 6: Check the final output sent to frontend ---
        console.log('\n--- DEBUG POINT 6: Final Recipes Sent to Frontend (Truncated URLs) ---');
        recipesWithImages.forEach(rec => {
            console.log(`Recipe ID: '${rec._id}', Title: '${rec.title}', ImageURL: ${rec.imageUrl ? rec.imageUrl.substring(0, 70) + '...' : 'null'}`);
        });
        console.log('----------------------------------------------------');*/

        if (llmResponseText.includes("I don't have enough information about recipes")) {
        recipesWithImages = [];
        }


        // 5. Send response to frontend
        res.json({
            llm_response: llmResponseText,
            retrieved_recipes: recipesWithImages
        });

    } catch (error) {
        console.error("Error during RAG search:", error);
        res.status(500).json({ error: "An error occurred during recipe search." });
    }
});

app.get('/api/recipes/:id', async (req, res) => {
    try {
        const recipeId = req.params.id;

        const recipe = await db.collection("recipes").findOne({ _id: recipeId });

        if (!recipe) {
            return res.status(404).json({ error: "Recipe not found in backend database." });
        }

        // Step 1: Format recipe input for LLM prompt
        const llmInput = {
            _id: recipe._id,
            title: recipe.title,
            ingredients: recipe.ingredients,
            instructions: recipe.instructions
        };

        const llmPrompt = `
            You are a structured recipe processor. You will be given raw recipe data in JSON format. 
            Return a cleaned and structured object optimized. 
            Assume the recipe is intended for 2–4 servings. 
            Estimate prep time and nutrition facts per serving based on standard culinary data. 
            Return the final response strictly raw object (no explanation, headers, or formatting hints).
            {
            "id": "string",                 // same as _id from original
            "title": "string",
            "prepTimeMinutes": number,     // estimated in minutes
            "servings": number,            // estimated between 2 and 4
            "ingredients": [ "string", ... ],
            "instructions": [ "string", ... ], //do not number these, just return as a list of clear steps
            "nutrition": {
                "perServing": {
                "calories": number,
                "totalFat": number,
                "saturatedFat": number,
                "carbohydrates": number,
                "sugar": number,
                "protein": number
                }
            }
            }
            Clean all fields for readability:
            Keep ingredients exactly as-is, unless there's a formatting issue (e.g., fix “34 cup sugar” to “3/4 cup sugar”).
            Make instructions a clear & understable sentences, numbered array. 
            All numbers are per-serving estimates, assuming the original recipe is meant for 2–4 people. 
            Now process this recipe input:
            ${JSON.stringify(llmInput)}`;

        let formattedRecipe;
        try {

            // --- LLM call proceeds ONLY if recipes are found ---
            const chat2 = chatModel.startChat({
                generationConfig: {
                    temperature: 0.1,
                    topK: 1,
                    topP: 1,
                },
                safetySettings: [
                    {
                        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold: HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold: HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold: HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold: HarmBlockThreshold.BLOCK_NONE,
                    },
                ],
            });

            // Prepare the context for the LLM
            const chatResult = await chat2.sendMessage(llmPrompt);
            const responseText = chatResult.response.text();
            let cleanedResponse = responseText;
            cleanedResponse = cleanedResponse.replace(/```json\n|\n```/g, '');
            cleanedResponse = cleanedResponse.trim();
            formattedRecipe = JSON.parse(cleanedResponse);

        } catch (llmError) {
            console.error("Error during LLM formatting:", llmError);
            return res.status(500).json({ error: "An error occurred during recipe formatting." });
        }

        try {
            const imageDoc = await db2.collection("recipe_images_full").findOne({ _id: recipeId });
            formattedRecipe.imageUrl = (imageDoc?.images?.[0]?.url) || null;

        } catch (imgError) {
            console.warn("Image URL fetch failed or missing:", imgError);
            formattedRecipe.imageUrl = null;
        }

        res.json(formattedRecipe);

    } catch (err) {
        console.error("Error in recipe detail API:", err);
        res.status(500).json({ error: "Internal server error while fetching recipe." });
    }
});

app.get('/api/summaries', async (req, res) => {
    try {
        const idsParam = req.query.ids;
        if (!idsParam) {
            return res.status(400).json({ error: "Missing 'ids' query parameter." });
        }
        const recipeIds = idsParam.split(',');

        const recipes = await db.collection("recipes").find(
            { _id: { $in: recipeIds } }, // Query for multiple IDs
            { projection: { _id: 1, title: 1, prepTimeMinutes: 1, servings: 1 } } // Only project necessary fields
        ).toArray();

        // 3. Fetch image URLs for all these recipes from the 'recipe_images_full' collection
        const imageCollection = db2.collection("recipe_images_full");
        const imageDocs = await imageCollection.find(
            { _id: { $in: recipeIds } }, // Query for multiple image IDs
            { projection: { _id: 1, images: { $slice: 1 } } } // Project _id and only the first image
        ).toArray();

        // Create a map for quick image URL lookup
        const imageUrlMap = new Map();
        imageDocs.forEach(doc => {
            if (doc.images && doc.images.length > 0 && doc.images[0].url) {
                imageUrlMap.set(doc._id.toString(), doc.images[0].url);
            }
        });

        // 4. Format the recipes and attach image URLs
        const formattedSummaries = recipes.map(recipe => {
            const recipeIdString = recipe._id.toString();
            const imageUrl = imageUrlMap.get(recipeIdString) || null;

            return {
                id: recipeIdString,
                title: recipe.title,
                prepTimeMinutes: recipe.prepTimeMinutes || null,
                servings: recipe.servings || null,
                imageUrl: imageUrl
            };
        });

        res.json(formattedSummaries);

    } catch (error) {
        console.error("Error fetching recipe summaries:", error);
        res.status(500).json({ error: "An error occurred while fetching recipe summaries." });
    }
});