import zipfile
from io import BytesIO

def predict_species_batch(image_folder, model, transform, classes, device, threshold=0.7):
    results = []
    for image_file in os.listdir(image_folder):
        try:
            image_path = os.path.join(image_folder, image_file)
            if not image_file.lower().endswith(('jpg', 'jpeg', 'png')):
                continue
            
            image = Image.open(image_path).convert("RGB")
            species, confidence, top_predictions, proc_time = predict_species(image, model, transform, classes, device, threshold)
            results.append({
                "file_name": image_file,
                "species": species,
                "confidence": confidence,
                "top_predictions": top_predictions,
                "processing_time": proc_time
            })
        except Exception as e:
            results.append({"file_name": image_file, "error": str(e)})
    return results

def main():
    # Existing main code...
    st.markdown('<div class="main-header"><h1>🦁 Species Recognition</h1></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("About")
        st.write("Upload an image or a folder of images to identify species using our deep learning model.")
        st.markdown("---")
        st.write("**Model Details:**")
        st.write("- CNN Architecture")
        st.write("- Input Size: 224x224")
        st.write("- Confidence Threshold: 70%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./Datasets/Train/animals"
    model_path = "./best_model.pth"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)

    with st.spinner("Loading model..."):
        model = load_model(model_path, num_classes, device)

    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    uploaded_zip = st.file_uploader("Upload a folder (as zip):", type=["zip"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            species, confidence, top_predictions, proc_time = predict_species(image, model, transform, classes, device)

            if top_predictions:
                st.success(f"Analysis complete! (Time: {proc_time:.2f}s)")
                for i, (pred_species, pred_conf) in enumerate(top_predictions, 1):
                    confidence_class = (
                        "confidence-high" if pred_conf > 0.7
                        else "confidence-medium" if pred_conf > 0.4
                        else "confidence-low"
                    )
                    st.markdown(
                        f"<div class='prediction-box'><b>#{i}</b>: {pred_species} <span class='{confidence_class}'>({pred_conf:.1%})</span></div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Confidence is too low to determine the species.")

    if uploaded_zip:
        with st.spinner("Processing folder..."):
            zip_path = os.path.join("./temp", "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            extract_dir = os.path.join("./temp", "extracted")
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            batch_results = predict_species_batch(extract_dir, model, transform, classes, device)

            st.success("Batch analysis complete!")
            for result in batch_results:
                if "error" in result:
                    st.error(f"File: {result['file_name']} - Error: {result['error']}")
                else:
                    st.markdown(f"**File:** {result['file_name']}")
                    for i, (pred_species, pred_conf) in enumerate(result["top_predictions"], 1):
                        confidence_class = (
                            "confidence-high" if pred_conf > 0.7
                            else "confidence-medium" if pred_conf > 0.4
                            else "confidence-low"
                        )
                        st.markdown(
                            f"<div class='prediction-box'><b>#{i}</b>: {pred_species} <span class='{confidence_class}'>({pred_conf:.1%})</span></div>",
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
