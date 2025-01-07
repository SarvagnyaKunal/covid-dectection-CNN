// X-ray file upload interaction
const xrayUpload = document.getElementById('xray-upload');
const xrayUploadArea = document.getElementById('xray-upload-area');
const xrayBrowseBtn = document.getElementById('xray-browse-btn');

xrayBrowseBtn.addEventListener('click', () => xrayUpload.click());

xrayUpload.addEventListener('change', () => {
  if (xrayUpload.files.length > 0) {
    xrayUploadArea.querySelector('p').textContent = `${xrayUpload.files.length} file(s) selected`;
  }
});

xrayUploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  xrayUploadArea.classList.add('drag-over');
});

xrayUploadArea.addEventListener('dragleave', () => {
  xrayUploadArea.classList.remove('drag-over');
});

xrayUploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  xrayUpload.files = e.dataTransfer.files;
  xrayUpload.dispatchEvent(new Event('change'));
});

// Reference documents upload interaction
const referenceUpload = document.getElementById('reference-upload');
const referenceUploadArea = document.getElementById('reference-upload-area');
const referenceBrowseBtn = document.getElementById('reference-browse-btn');

referenceBrowseBtn.addEventListener('click', () => referenceUpload.click());

referenceUpload.addEventListener('change', () => {
  if (referenceUpload.files.length > 0) {
    referenceUploadArea.querySelector('p').textContent = `${referenceUpload.files.length} file(s) selected`;
  }
});

referenceUploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  referenceUploadArea.classList.add('drag-over');
});

referenceUploadArea.addEventListener('dragleave', () => {
  referenceUploadArea.classList.remove('drag-over');
});

referenceUploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  referenceUpload.files = e.dataTransfer.files;
  referenceUpload.dispatchEvent(new Event('change'));
});
