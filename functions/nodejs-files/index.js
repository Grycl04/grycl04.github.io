const functions = require("firebase-functions");
const {onCall} = require("firebase-functions/v2/https");
const {onDocumentUpdated} = require("firebase-functions/v2/firestore");
const admin = require("firebase-admin");
const axios = require("axios");
const FormData = require("form-data");
const {getStorage} = require('firebase-admin/storage');

admin.initializeApp();

// Face++ API Configuration
require("dotenv").config();
const FACE_PP_API_KEY = process.env.FACE_PP_API_KEY;
const FACE_PP_API_SECRET = process.env.FACE_PP_API_SECRET;

// Face++ API Endpoints
const FACE_PP_DETECT_URL = "https://api-us.faceplusplus.com/facepp/v3/detect";
const FACE_PP_COMPARE_URL = "https://api-us.faceplusplus.com/facepp/v3/compare";
const FACE_PP_OCR_URL = "https://api-us.faceplusplus.com/cardpp/v1/ocridcard";

const CONFIDENCE_THRESHOLD = 65;

/**
 * Delete KYC files from Storage
 */
async function deleteKYCFiles(userId) {
  try {
    const bucket = getStorage().bucket();
    const prefix = `kyc/${userId}/`;
    
    console.log(`Deleting files with prefix: ${prefix}`);
    
    const [files] = await bucket.getFiles({prefix: prefix});
    
    if (files.length === 0) {
      console.log('No files to delete');
      return;
    }
    
    await Promise.all(files.map(file => file.delete()));
    
    console.log(`Deleted ${files.length} files for user ${userId}`);
  } catch (error) {
    console.error('Error deleting KYC files:', error);
  }
}

/**
 * Step 1: Detect face in ID photo using URL
 */
async function detectFaceInID(idImageUrl) {
  try {
    console.log("Detecting face in ID from URL...");
    const formData = new FormData();
    formData.append("api_key", FACE_PP_API_KEY);
    formData.append("api_secret", FACE_PP_API_SECRET);
    formData.append("image_url", idImageUrl);
    formData.append("return_attributes", "gender,age,blur,eyestatus");

    const response = await axios.post(FACE_PP_DETECT_URL, formData, {
      headers: formData.getHeaders(),
      timeout: 30000,
    });

    console.log("ID Face Detection Response:", JSON.stringify(response.data, null, 2));

    if (response.data.faces && response.data.faces.length > 0) {
      const face = response.data.faces[0];
      const blur = face.attributes?.blur;
      
      if (blur && blur.blurness && blur.blurness.value > 50) {
        console.log(`Warning: ID image is blurry (${blur.blurness.value})`);
      }

      return {
        success: true,
        faceToken: face.face_token,
        attributes: face.attributes,
        faceCount: response.data.faces.length,
        quality: {
          blur: blur?.blurness?.value || 0,
          threshold: blur?.blurness?.threshold || 0
        }
      };
    }

    return {
      success: false,
      error: "No face detected in ID photo. Please upload a clear photo of your ID with visible face."
    };
  } catch (error) {
    console.error("Error detecting face in ID:", error.response?.data || error.message);
    const errorMsg = error.response?.data?.error_message || error.message;

    if (errorMsg.includes("INVALID_IMAGE_URL")) {
      return {success: false, error: "Unable to access ID image. Please try uploading again."};
    }
    if (errorMsg.includes("IMAGE_ERROR")) {
      return {success: false, error: "ID image format error. Please ensure it's a clear JPG or PNG."};
    }

    return {success: false, error: `Face detection failed: ${errorMsg}`};
  }
}

/**
 * Step 2: Detect face in selfie with retry logic
 */
async function detectFaceInSelfie(selfieImageUrl, retries = 3) {
  try {
    console.log("Detecting face in selfie from URL...");
    const formData = new FormData();
    formData.append("api_key", FACE_PP_API_KEY);
    formData.append("api_secret", FACE_PP_API_SECRET);
    formData.append("image_url", selfieImageUrl);
    formData.append("return_attributes", "gender,age,blur,eyestatus");

    const response = await axios.post(FACE_PP_DETECT_URL, formData, {
      headers: formData.getHeaders(),
      timeout: 30000,
    });

    console.log("Selfie Face Detection Response:", JSON.stringify(response.data, null, 2));

    if (response.data.faces && response.data.faces.length > 0) {
      if (response.data.faces.length > 1) {
        console.log(`Warning: ${response.data.faces.length} faces detected, using largest face`);
      }

      const faces = response.data.faces.sort((a, b) => {
        const areaA = a.face_rectangle.width * a.face_rectangle.height;
        const areaB = b.face_rectangle.width * b.face_rectangle.height;
        return areaB - areaA;
      });

      const face = faces[0];
      const blur = face.attributes?.blur;
      
      if (blur && blur.blurness && blur.blurness.value > 50) {
        console.log(`Warning: Selfie is blurry (${blur.blurness.value})`);
      }

      return {
        success: true,
        faceToken: face.face_token,
        attributes: face.attributes,
        quality: {
          blur: blur?.blurness?.value || 0,
          threshold: blur?.blurness?.threshold || 0
        }
      };
    }

    return {
      success: false,
      error: "No face detected in selfie. Please take a clear selfie with your face fully visible."
    };
  } catch (error) {
    const errorMsg = error.response?.data?.error_message || error.message;
    console.error("Error detecting face in selfie:", errorMsg);

    if (errorMsg.includes("CONCURRENCY_LIMIT_EXCEEDED") && retries > 0) {
      console.log(`Rate limited, retrying in 2 seconds... (${retries} retries left)`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      return detectFaceInSelfie(selfieImageUrl, retries - 1);
    }

    if (errorMsg.includes("INVALID_IMAGE_URL")) {
      return {success: false, error: "Unable to access selfie image. Please try uploading again."};
    }

    return {success: false, error: `Face detection failed: ${errorMsg}`};
  }
}

/**
 * Step 3: Compare faces using face tokens
 */
async function compareFaces(idFaceToken, selfieFaceToken, retries = 3) {
  try {
    console.log("Comparing face tokens...");
    const formData = new FormData();
    formData.append("api_key", FACE_PP_API_KEY);
    formData.append("api_secret", FACE_PP_API_SECRET);
    formData.append("face_token1", idFaceToken);
    formData.append("face_token2", selfieFaceToken);

    const response = await axios.post(FACE_PP_COMPARE_URL, formData, {
      headers: formData.getHeaders(),
      timeout: 30000,
    });

    console.log("Face Compare Response:", JSON.stringify(response.data, null, 2));

    if (response.data.confidence !== undefined) {
      return {
        success: true,
        confidence: response.data.confidence,
        thresholds: response.data.thresholds,
      };
    }

    return {
      success: false,
      error: "Face comparison failed. Please ensure your photos are clear and show your face."
    };
  } catch (error) {
    const errorMsg = error.response?.data?.error_message || error.message;
    console.error("Error comparing faces:", errorMsg);

    if (errorMsg.includes("CONCURRENCY_LIMIT_EXCEEDED") && retries > 0) {
      console.log(`Rate limited, retrying in 2 seconds... (${retries} retries left)`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      return compareFaces(idFaceToken, selfieFaceToken, retries - 1);
    }

    return {success: false, error: `Face comparison failed: ${errorMsg}`};
  }
}

/**
 * Step 4: OCR ID Card (Extract text from ID) - Optional
 */
async function extractIDData(idImageUrl) {
  try {
    console.log("Extracting ID data via OCR...");
    const formData = new FormData();
    formData.append("api_key", FACE_PP_API_KEY);
    formData.append("api_secret", FACE_PP_API_SECRET);
    formData.append("image_url", idImageUrl);

    const response = await axios.post(FACE_PP_OCR_URL, formData, {
      headers: formData.getHeaders(),
      timeout: 30000,
    });

    console.log("OCR Response:", JSON.stringify(response.data, null, 2));
    return {success: true, data: response.data};
  } catch (error) {
    console.error("Error extracting ID data:", error.response?.data || error.message);
    return {
      success: false,
      error: error.response?.data?.error_message || "ID extraction failed"
    };
  }
}

/**
 * Main Cloud Function: Verify KYC Documents with Auto-Delete on Rejection
 */
exports.verifyKYC = onCall(async (request) => {
  if (!request.auth) {
    throw new functions.https.HttpsError(
      "unauthenticated",
      "User must be authenticated to verify KYC"
    );
  }

  const userId = request.auth.uid;
  const {governmentIdUrl, selfieWithIdUrl} = request.data;

  if (!governmentIdUrl || !selfieWithIdUrl) {
    throw new functions.https.HttpsError(
      "invalid-argument",
      "Government ID URL and Selfie URL are required"
    );
  }

  try {
    console.log(`========== Starting KYC verification for user: ${userId} ==========`);
    console.log(`ID URL: ${governmentIdUrl.substring(0, 80)}...`);
    console.log(`Selfie URL: ${selfieWithIdUrl.substring(0, 80)}...`);

    // Step 1: Detect face in ID
    console.log("Step 1: Detecting face in ID...");
    const idFaceResult = await detectFaceInID(governmentIdUrl);

    if (!idFaceResult.success) {
      console.log("Step 1 Failed:", idFaceResult.error);
      
      // DELETE FILES BEFORE UPDATING FIRESTORE
      await deleteKYCFiles(userId);
      
      await admin.firestore().collection("users").doc(userId).update({
        kycStatus: "rejected",
        kycRejectionReason: idFaceResult.error,
        kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
        kycDocuments: admin.firestore.FieldValue.delete(),
      });

      return {
        success: false,
        status: "rejected",
        reason: idFaceResult.error,
        details: {step: "id_face_detection", error: idFaceResult.error},
      };
    }

    console.log("Step 1 Success: Face detected in ID");
    console.log(`Quality - Blur: ${idFaceResult.quality.blur.toFixed(2)}`);

    // Step 2: Detect face in selfie
    console.log("Step 2: Detecting face in selfie...");
    const selfieFaceResult = await detectFaceInSelfie(selfieWithIdUrl);

    if (!selfieFaceResult.success) {
      console.log("Step 2 Failed:", selfieFaceResult.error);
      
      // DELETE FILES
      await deleteKYCFiles(userId);
      
      await admin.firestore().collection("users").doc(userId).update({
        kycStatus: "rejected",
        kycRejectionReason: selfieFaceResult.error,
        kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
        kycDocuments: admin.firestore.FieldValue.delete(),
      });

      return {
        success: false,
        status: "rejected",
        reason: selfieFaceResult.error,
        details: {step: "selfie_face_detection", error: selfieFaceResult.error},
      };
    }

    console.log("Step 2 Success: Face detected in selfie");
    console.log(`Quality - Blur: ${selfieFaceResult.quality.blur.toFixed(2)}`);

    // Step 3: Compare faces
    console.log("Step 3: Comparing faces...");
    const compareResult = await compareFaces(
      idFaceResult.faceToken,
      selfieFaceResult.faceToken
    );

    if (!compareResult.success) {
      console.log("Step 3 Failed:", compareResult.error);
      
      // DELETE FILES
      await deleteKYCFiles(userId);
      
      await admin.firestore().collection("users").doc(userId).update({
        kycStatus: "rejected",
        kycRejectionReason: compareResult.error,
        kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
        kycDocuments: admin.firestore.FieldValue.delete(),
      });

      return {
        success: false,
        status: "rejected",
        reason: compareResult.error,
        details: {step: "face_comparison", error: compareResult.error},
      };
    }

    const confidence = compareResult.confidence;
    console.log(`Step 3 Success: Face match confidence: ${confidence.toFixed(2)}%`);

    // Check confidence threshold
    if (confidence < CONFIDENCE_THRESHOLD) {
      const reason = `Face match confidence too low: ${confidence.toFixed(2)}% (minimum required: ${CONFIDENCE_THRESHOLD}%)`;
      console.log("Verification Failed:", reason);
      
      // DELETE FILES
      await deleteKYCFiles(userId);
      
      await admin.firestore().collection("users").doc(userId).update({
        kycStatus: "rejected",
        kycRejectionReason: reason,
        kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
        kycDocuments: admin.firestore.FieldValue.delete(),
        kycVerificationDetails: {
          faceMatchConfidence: confidence,
          threshold: CONFIDENCE_THRESHOLD,
          idQuality: idFaceResult.quality,
          selfieQuality: selfieFaceResult.quality,
        }
      });

      return {
        success: false,
        status: "rejected",
        reason: reason,
        details: {
          step: "face_comparison",
          confidence: confidence,
          threshold: CONFIDENCE_THRESHOLD,
        },
      };
    }

    // Step 4: Extract ID data (optional)
    console.log("Step 4: Extracting ID data (optional)...");
    const ocrResult = await extractIDData(governmentIdUrl);

    if (ocrResult.success) {
      console.log("Step 4 Success: ID data extracted");
    } else {
      console.log("Step 4 Warning: ID extraction failed (non-critical)");
    }

    // All checks passed - Approve KYC (KEEP FILES)
    console.log(`========== KYC APPROVED for user: ${userId} ==========`);
    
    await admin.firestore().collection("users").doc(userId).update({
      kycStatus: "approved",
      kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
      kycVerificationDetails: {
        faceMatchConfidence: confidence,
        idFaceDetected: true,
        selfieFaceDetected: true,
        idQuality: idFaceResult.quality,
        selfieQuality: selfieFaceResult.quality,
        ocrExtracted: ocrResult.success,
        verifiedAt: new Date().toISOString(),
      },
    });

    return {
      success: true,
      status: "approved",
      message: "KYC verification successful! Your account has been verified.",
      details: {
        confidence: confidence,
        threshold: CONFIDENCE_THRESHOLD,
        ocrData: ocrResult.success ? ocrResult.data : null,
      },
    };

  } catch (error) {
    console.error("========== KYC Verification Error ==========");
    console.error(error);
    
    // DELETE FILES on technical error
    await deleteKYCFiles(userId);
    
    await admin.firestore().collection("users").doc(userId).update({
      kycStatus: "rejected",
      kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
      kycRejectionReason: `Technical error: ${error.message}`,
      kycDocuments: admin.firestore.FieldValue.delete(),
    });

    throw new functions.https.HttpsError(
      "internal",
      "KYC verification failed due to technical error",
      error.message
    );
  }
});

/**
 * Firestore Trigger: Auto-verify when KYC documents are submitted
 */
exports.onKYCSubmitted = onDocumentUpdated("users/{userId}", async (event) => {
  const before = event.data.before.data();
  const after = event.data.after.data();
  const userId = event.params.userId;

  if (before && after && before.kycStatus !== "pending" && after.kycStatus === "pending") {
    console.log(`========== Auto-triggering KYC verification for user: ${userId} ==========`);
    
    try {
      const kycDocs = after.kycDocuments;
      
      if (!kycDocs || !kycDocs.governmentId || !kycDocs.selfieWithId) {
        console.log("Missing required documents, skipping auto-verification");
        return null;
      }

      const result = await verifyKYCInternal(
        userId,
        kycDocs.governmentId,
        kycDocs.selfieWithId
      );

      console.log(`Auto-verification result for ${userId}:`, result);
    } catch (error) {
      console.error(`Auto-verification error for ${userId}:`, error);
    }
  }

  return null;
});

/**
 * Internal verification function with auto-delete
 */
async function verifyKYCInternal(userId, governmentIdUrl, selfieWithIdUrl) {
  try {
    console.log(`Internal verification starting for ${userId}`);

    // Step 1: Detect face in ID
    const idFaceResult = await detectFaceInID(governmentIdUrl);
    if (!idFaceResult.success) {
      await deleteKYCFiles(userId);
      await admin.firestore().collection("users").doc(userId).update({
        kycStatus: "rejected",
        kycRejectionReason: idFaceResult.error,
        kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
        kycDocuments: admin.firestore.FieldValue.delete(),
      });
      return {success: false, reason: idFaceResult.error};
    }

    // Step 2: Detect face in selfie
    const selfieFaceResult = await detectFaceInSelfie(selfieWithIdUrl);
    if (!selfieFaceResult.success) {
      await deleteKYCFiles(userId);
      await admin.firestore().collection("users").doc(userId).update({
        kycStatus: "rejected",
        kycRejectionReason: selfieFaceResult.error,
        kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
        kycDocuments: admin.firestore.FieldValue.delete(),
      });
      return {success: false, reason: selfieFaceResult.error};
    }

    // Step 3: Compare faces
    const compareResult = await compareFaces(
      idFaceResult.faceToken,
      selfieFaceResult.faceToken
    );

    if (!compareResult.success) {
      await deleteKYCFiles(userId);
      await admin.firestore().collection("users").doc(userId).update({
        kycStatus: "rejected",
        kycRejectionReason: compareResult.error,
        kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
        kycDocuments: admin.firestore.FieldValue.delete(),
      });
      return {success: false, reason: compareResult.error};
    }

    const confidence = compareResult.confidence;
    if (confidence < CONFIDENCE_THRESHOLD) {
      const reason = `Face match confidence too low: ${confidence.toFixed(2)}%`;
      await deleteKYCFiles(userId);
      await admin.firestore().collection("users").doc(userId).update({
        kycStatus: "rejected",
        kycRejectionReason: reason,
        kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
        kycDocuments: admin.firestore.FieldValue.delete(),
      });
      return {success: false, reason: reason};
    }

    // Approve - keep files
    await admin.firestore().collection("users").doc(userId).update({
      kycStatus: "approved",
      kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
      kycVerificationDetails: {
        faceMatchConfidence: confidence,
        idQuality: idFaceResult.quality,
        selfieQuality: selfieFaceResult.quality,
        verifiedAt: new Date().toISOString(),
      },
    });

    return {success: true, confidence: confidence};

  } catch (error) {
    console.error("Internal verification error:", error);
    await deleteKYCFiles(userId);
    await admin.firestore().collection("users").doc(userId).update({
      kycStatus: "rejected",
      kycRejectionReason: `Technical error: ${error.message}`,
      kycVerificationDate: admin.firestore.FieldValue.serverTimestamp(),
      kycDocuments: admin.firestore.FieldValue.delete(),
    });
    return {success: false, reason: "Technical error"};
  }
}