document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();
  
    const imageInput = document.getElementById('imageInput');
    const loadingMessage = document.getElementById('loadingMessage');
    const resultSection = document.getElementById('resultSection');
    const uploadedImage = document.getElementById('uploadedImage');
    const speciesName = document.getElementById('speciesName');
  
    if (imageInput.files.length === 0) {
      alert('Please select an image to upload.');
      return;
    }
  
    // Show loading message
    loadingMessage.style.display = 'block';
    resultSection.style.display = 'none';
  
    // Simulate API response
    const fakeApiResponse = { species: 'Tiger' };
  
    setTimeout(() => {
      // Hide loading message
      loadingMessage.style.display = 'none';
  
      // Show results
      resultSection.style.display = 'block';
      uploadedImage.src = URL.createObjectURL(imageInput.files[0]);
      speciesName.textContent = fakeApiResponse.species;
    }, 2000);
  });
  
  // Smooth Scrolling for Navigation Links
  document.querySelectorAll('nav a').forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('href').substring(1);
      const targetSection = document.getElementById(targetId);
      targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
  
  
  