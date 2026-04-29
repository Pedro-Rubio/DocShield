import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, ActivityIndicator } from 'react-native';
import { Camera, useCameraDevices, useFrameProcessor } from 'react-native-vision-camera';
import { useTensorFlow } from '@tensorflow/tfjs-react-native'; // Opcional para procesamiento local
import * as ImageManipulator from 'expo-image-manipulator';
import * as Sensors from 'expo-sensors';
import { Accelerometer } from 'expo-sensors';
import { captureDocument } from './utils/captureUtils'; // Implementar según necesidad

interface DocumentCaptureProps {
  onVerificationResult: (result: any) => void;
  apiEndpoint?: string;
}

export default function DocumentCapture({ 
  onVerificationResult, 
  apiEndpoint = 'http://localhost:8000/api/v1/verify-document' 
}: DocumentCaptureProps) {
  const [hasPermission, setHasPermission] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [livenessPassed, setLivenessPassed] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [sessionId] = useState(() => Math.random().toString(36).substring(7));
  
  const devices = useCameraDevices();
  const device = devices.back;
  const cameraRef = useRef(null);
  
  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'authorized');
      
      // Configurar acelerómetro para liveness detection
      Accelerometer.setUpdateInterval(100);
      const subscription = Accelerometer.addListener(accelerometerData => {
        const { x, y, z } = accelerometerData;
        const currentRotation = Math.atan2(y, x) * (180 / Math.PI);
        setRotation(Math.abs(currentRotation));
        
        // Detectar inclinación de ±10° (liveness)
        if (Math.abs(currentRotation) > 10) {
          setLivenessPassed(true);
        }
      });
      
      return () => subscription.remove();
    })();
  }, []);
  
  const captureAndVerify = async () => {
    if (!cameraRef.current || isCapturing) return;
    
    setIsCapturing(true);
    
    try {
      // Capturar imagen
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'quality',
        flash: 'off',
        enableAutoRedEyeReduction: true
      });
      
      // Comprimir en memoria (NO guardar en disco)
      const manipResult = await ImageManipulator.manipulateAsync(
        photo.path,
        [{ resize: { width: 1200 } }],
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG, base64: true }
      );
      
      if (!manipResult.base64) {
        throw new Error('No se pudo obtener base64 de la imagen');
      }
      
      // Preparar metadatos de captura
      const captureMeta = {
        timestamp: new Date().toISOString(),
        device_model: Platform.OS === 'ios' ? 'iOS' : 'Android',
        liveness_passed: livenessPassed,
        liveness_rotation_deg: rotation,
        session_id: sessionId,
        accelerometer_data: [rotation] // Simplificado
      };
      
      // Enviar a API para verificación
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: manipResult.base64,
          capture_meta: captureMeta
        })
      });
      
      const result = await response.json();
      
      // Eliminar base64 de memoria (no hay forma directa, pero no se guarda)
      onVerificationResult(result);
      
    } catch (error) {
      Alert.alert('Error', `Error en captura: ${error.message}`);
    } finally {
      setIsCapturing(false);
    }
  };
  
  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Se requiere permiso de cámara</Text>
      </View>
    );
  }
  
  if (!device) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Cámara no disponible</Text>
      </View>
    );
  }
  
  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={styles.camera}
        device={device}
        isActive={true}
        photo={true}
      />
      
      {/* Marco guía para documento */}
      <View style={styles.guideFrame}>
        <View style={styles.cornerTopLeft} />
        <View style={styles.cornerTopRight} />
        <View style={styles.cornerBottomLeft} />
        <View style={styles.cornerBottomRight} />
      </View>
      
      <View style={styles.controls}>
        {!livenessPassed && (
          <Text style={styles.livenessText}>
            Incliná el documento ±10° para verificación
          </Text>
        )}
        {livenessPassed && (
          <Text style={styles.livenessSuccess}>
            ✓ Liveness detectado
          </Text>
        )}
        
        <TouchableOpacity
          style={[styles.captureButton, isCapturing && styles.capturingButton]}
          onPress={captureAndVerify}
          disabled={isCapturing}
        >
          {isCapturing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.captureButtonText}>Capturar Documento</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  guideFrame: {
    position: 'absolute',
    top: '20%',
    left: '10%',
    right: '10%',
    bottom: '30%',
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.7)',
    borderRadius: 10,
  },
  cornerTopLeft: {
    position: 'absolute',
    top: -2,
    left: -2,
    width: 20,
    height: 20,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderColor: '#00FF00',
  },
  cornerTopRight: {
    position: 'absolute',
    top: -2,
    right: -2,
    width: 20,
    height: 20,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderColor: '#00FF00',
  },
  cornerBottomLeft: {
    position: 'absolute',
    bottom: -2,
    left: -2,
    width: 20,
    height: 20,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderColor: '#00FF00',
  },
  cornerBottomRight: {
    position: 'absolute',
    bottom: -2,
    right: -2,
    width: 20,
    height: 20,
    borderBottomWidth: 4,
    borderRightWidth: 4,
    borderColor: '#00FF00',
  },
  controls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  livenessText: {
    color: '#FFA500',
    marginBottom: 10,
    textAlign: 'center',
  },
  livenessSuccess: {
    color: '#00FF00',
    marginBottom: 10,
    textAlign: 'center',
  },
  captureButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 30,
    minWidth: 200,
    alignItems: 'center',
  },
  capturingButton: {
    backgroundColor: '#FF3B30',
  },
  captureButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  text: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 50,
  },
});
