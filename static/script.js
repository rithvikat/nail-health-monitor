// Nail Health Monitor - Simple Frontend Script

document.addEventListener("DOMContentLoaded", () => {
  // 1. File Upload Preview & Drag-and-Drop
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("image-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const fileNameDisplay = document.getElementById("file-name");
  const submitBtn = document.getElementById("submit-btn");

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      handleSelectedFile(file);
    });
  }

  if (dropzone) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        fileInput.files = e.dataTransfer.files;
        handleSelectedFile(e.dataTransfer.files[0]);
      }
    });
  }

  function handleSelectedFile(file) {
    if (!file) return;

    // Check if it's an image
    if (!file.type.startsWith("image/")) {
      alert("Please choose an image file (JPEG, PNG, or WebP).");
      return;
    }

    if (fileNameDisplay) {
      fileNameDisplay.textContent = `Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`;
    }

    // Read and display preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImage) {
        previewImage.src = e.target.result;
      }
      if (previewContainer) {
        previewContainer.style.display = "block";
      }
      if (submitBtn) {
        submitBtn.disabled = false;
      }
    };
    reader.readAsDataURL(file);
  }

  // 2. Submit Button Loading State
  const screeningForm = document.getElementById("screening-form");
  if (screeningForm && submitBtn) {
    screeningForm.addEventListener("submit", () => {
      submitBtn.disabled = true;
      submitBtn.innerHTML = "Analyzing Image...";
    });
  }
});
