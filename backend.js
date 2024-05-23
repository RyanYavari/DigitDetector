const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const cors = require('cors');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors()); // Enable CORS for all routes

app.post('/upload', upload.single('file'), async (req, res) => {
    let filePath;
    try {
        console.log('Received file:', req.file);
        filePath = path.join(__dirname, req.file.path);
        console.log('File path:', filePath);

        // Convert the image to MNIST-like format (28x28, white text on black background)
        const destinationPath = path.join(__dirname, 'saved_images', req.file.originalname);
        const paddedImage = await sharp(filePath)
            // .extend({ top: 20, bottom: 10, left: 10, right: 10, background: { r: 0, g: 0, b: 0 } }) // Add 20 pixels of padding at the top and 10 pixels at the bottom, left, and right
            .resize(28, 28)
            .grayscale()
            .negate({ alpha: false }) // Invert colors to make text white on black background
            // .negate({ alpha: false })
            .toFile(destinationPath);

        console.log('File saved to:', destinationPath);

        // Read the processed file to send to the Python backend
        const fileBuffer = fs.readFileSync(destinationPath);

        // Create a form data object
        const formData = new FormData();
        formData.append('file', fileBuffer, req.file.originalname);

        // Send the file to the Python backend for prediction
        const predictResponse = await axios.post('http://localhost:5001/predict', formData, {
            headers: {
                ...formData.getHeaders()
            }
        });

        console.log('Prediction received:', predictResponse.data);

        // Respond with the prediction result
        res.json(predictResponse.data);
    } catch (error) {
        console.error('Error uploading the drawing:', error);
        res.status(500).json({ error: 'Error uploading the drawing' });
    } finally {
        if (filePath) {
            fs.unlinkSync(filePath);
        }
    }
});

app.listen(5000, () => {
    console.log('Node.js server listening on port 5000');
});