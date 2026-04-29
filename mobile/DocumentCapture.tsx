/**
 * DocShield — Componente de captura de documento
 *
 * Componente React Native para captura de documentos de identidad
 * con detección de liveness mediante el acelerómetro del dispositivo.
 *
 * Principio: NUNCA permite subir imágenes, solo captura desde cámara.
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  Alert,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
} from "react-native";
import { Camera, useCameraDevice, useCameraFormat } from "react-native-vision-camera";
import { Accelerometer } from "expo-sensors";

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");

// Configuración
const LIVENESS_TIMEOUT_MS = 8000; // 8 segundos para liveness
const LIVENESS_ANGLE_DEG = 10; // +/- 10 grados de inclinación requeridos
const CAPTURE_MARGIN = 40; // margen del marco guía

type CaptureState = "idle" | "capturing" | "liveness" | "success" | "error" | "timeout";

interface DocumentCaptureProps {
  onCapture: (base64Image: string, metadata: CaptureMetadata) => void;
  onError: (error: string) => void;
}

interface CaptureMetadata {
  user_agent: string;
  screen_width: number;
  screen_height: number;
  platform: string;
  ip_address: string;
  liveness_passed: boolean;
  accelerometer_data: number[];
}

interface AccelerometerData {
  x: number;
  y: number;
  z: number;
}

const DocumentCapture: React.FC<DocumentCaptureProps> = ({ onCapture, onError }) => {
  const [state, setState] = useState<CaptureState>("idle");
  const [permission, setPermission] = useState<boolean>(false);
  const [livenessProgress, setLivenessProgress] = useState<number>(0);
  const [accelData, setAccelData] = useState<AccelerometerData>({ x: 0, y: 0, z: 0 });
  const [livenessPassed, setLivenessPassed] = useState<boolean>(false);

  const cameraRef = useRef<Camera>(null);
  const accelSubscription = useRef<any>(null);
  const livenessStartTime = useRef<number>(0);
  const accelReadings = useRef<number[]>([]);
  const maxAngleReached = useRef<number>(0);

  const device = useCameraDevice("back");
  const format = useCameraFormat(device, [
    { photoResolution: { width: 1920, height: 1080 } },
    { fps: 30 },
  ]);

  useEffect(() => {
    requestPermissions();
    return () => {
      accelSubscription.current?.remove();
    };
  }, []);

  const requestPermissions = async () => {
    try {
      const cameraPermission = await Camera.requestCameraPermission();
      setPermission(cameraPermission === "authorized");

      if (cameraPermission !== "authorized") {
        Alert.alert(
          "Permiso requerido",
          "DocShield necesita acceso a la cámara para capturar el documento."
        );
      }
    } catch (error) {
      onError("No se pudo solicitar permiso de cámara");
    }
  };

  const startLivenessDetection = useCallback(() => {
    setState("liveness");
    livenessStartTime.current = Date.now();
    accelReadings.current = [];
    maxAngleReached.current = 0;

    accelSubscription.current = Accelerometer.addListener((data) => {
      const { x, y, z } = data;
      setAccelData(data);
      accelReadings.current.push(Math.sqrt(x * x + y * y + z * z));

      // Calcular ángulo de inclinación
      const angle = Math.atan2(Math.sqrt(x * x + y * y), z) * (180 / Math.PI);
      maxAngleReached.current = Math.max(maxAngleReached.current, Math.abs(angle));

      // Verificar si alcanzó el ángulo requerido
      if (Math.abs(angle) >= LIVENESS_ANGLE_DEG) {
        setLivenessPassed(true);
      }

      // Actualizar progreso
      const elapsed = Date.now() - livenessStartTime.current;
      const progress = Math.min(elapsed / LIVENESS_TIMEOUT_MS, 1);
      setLivenessProgress(progress);

      // Timeout
      if (elapsed >= LIVENESS_TIMEOUT_MS) {
        stopAccelerometer();
        if (!livenessPassed) {
          setState("timeout");
          onError("Tiempo agotado. Debés inclinar el documento para verificar que es físico.");
        }
      }
    });

    Accelerometer.setUpdateInterval(100); // 10Hz
  }, [onError, livenessPassed]);

  const stopAccelerometer = useCallback(() => {
    accelSubscription.current?.remove();
    accelSubscription.current = null;
    Accelerometer.removeAllListeners();
  }, []);

  const handleCapture = async () => {
    if (!cameraRef.current || !permission) return;

    setState("capturing");
    startLivenessDetection();
  };

  const handleFinalCapture = async () => {
    if (!cameraRef.current) return;

    try {
      const photo = await cameraRef.current.takePhoto({
        flash: "off",
        qualityPrioritization: "quality",
      });

      // Leer archivo como base64
      const fileUri = `file://${photo.path}`;
      const response = await fetch(fileUri);
      const blob = await response.blob();

      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(",")[1];

        const metadata: CaptureMetadata = {
          user_agent: "DocShield-Mobile/1.0",
          screen_width: SCREEN_WIDTH,
          screen_height: SCREEN_HEIGHT,
          platform: Platform.OS,
          ip_address: "0.0.0.0", // Obtener de la API en producción
          liveness_passed: livenessPassed,
          accelerometer_data: accelReadings.current.slice(0, 80), // Limitar a 80 lecturas
        };

        stopAccelerometer();
        setState("success");
        onCapture(base64, metadata);
      };
    } catch (error) {
      stopAccelerometer();
      setState("error");
      onError("Error al capturar la imagen");
    }
  };

  const renderGuideOverlay = () => (
    <View style={styles.guideOverlay} pointerEvents="none">
      <View style={styles.guideBox}>
        <View style={styles.cornerTopLeft} />
        <View style={styles.cornerTopRight} />
        <View style={styles.cornerBottomLeft} />
        <View style={styles.cornerBottomRight} />
      </View>
      <Text style={styles.guideText}>
        {state === "liveness"
          ? livenessPassed
            ? "¡Bien! Ahora mantené el documento quieto"
            : "Incliná el documento ±10° para verificar"
          : "Alineá el documento dentro del marco"}
      </Text>
    </View>
  );

  const renderLivenessProgress = () => {
    if (state !== "liveness") return null;

    return (
      <View style={styles.livenessContainer}>
        <View style={styles.livenessBar}>
          <View
            style={[
              styles.livenessProgress,
              { width: `${livenessProgress * 100}%` },
            ]}
          />
        </View>
        <Text style={styles.livenessText}>
          {Math.max(0, Math.ceil((LIVENESS_TIMEOUT_MS - (Date.now() - livenessStartTime.current)) / 1000))}s
        </Text>
      </View>
    );
  };

  if (!device) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No se encontró cámara</Text>
      </View>
    );
  }

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Solicitando permiso de cámara...</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermissions}>
          <Text style={styles.buttonText}>Activar cámara</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={state !== "error"}
        photo={true}
        format={format}
      />

      {renderGuideOverlay()}
      {renderLivenessProgress()}

      {state === "idle" && (
        <View style={styles.bottomControls}>
          <TouchableOpacity style={styles.captureButton} onPress={handleCapture}>
            <Text style={styles.captureButtonText}>Capturar</Text>
          </TouchableOpacity>
        </View>
      )}

      {state === "liveness" && livenessPassed && (
        <View style={styles.bottomControls}>
          <TouchableOpacity style={styles.captureButton} onPress={handleFinalCapture}>
            <ActivityIndicator color="white" />
            <Text style={styles.captureButtonText}>Verificando...</Text>
          </TouchableOpacity>
        </View>
      )}

      {state === "capturing" && (
        <View style={styles.overlay}>
          <ActivityIndicator size="large" color="white" />
          <Text style={styles.overlayText}>Capturando documento...</Text>
        </View>
      )}

      {state === "timeout" && (
        <View style={styles.overlay}>
          <Text style={styles.errorText}>Tiempo agotado</Text>
          <TouchableOpacity style={styles.retryButton} onPress={() => setState("idle")}>
            <Text style={styles.buttonText}>Reintentar</Text>
          </TouchableOpacity>
        </View>
      )}

      {state === "error" && (
        <View style={styles.overlay}>
          <Text style={styles.errorText}>Error en la captura</Text>
          <TouchableOpacity style={styles.retryButton} onPress={() => setState("idle")}>
            <Text style={styles.buttonText}>Reintentar</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "black",
  },
  text: {
    color: "white",
    fontSize: 18,
    textAlign: "center",
    marginTop: 50,
  },
  errorText: {
    color: "#ff4444",
    fontSize: 18,
    textAlign: "center",
    fontWeight: "bold",
  },
  guideOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "center",
    alignItems: "center",
  },
  guideBox: {
    width: SCREEN_WIDTH - CAPTURE_MARGIN * 2,
    height: (SCREEN_WIDTH - CAPTURE_MARGIN * 2) * 1.6,
    borderWidth: 2,
    borderColor: "rgba(255, 255, 255, 0.5)",
    borderRadius: 12,
  },
  cornerTopLeft: {
    position: "absolute",
    top: -2,
    left: -2,
    width: 40,
    height: 40,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderColor: "#4CAF50",
    borderTopLeftRadius: 8,
  },
  cornerTopRight: {
    position: "absolute",
    top: -2,
    right: -2,
    width: 40,
    height: 40,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderColor: "#4CAF50",
    borderTopRightRadius: 8,
  },
  cornerBottomLeft: {
    position: "absolute",
    bottom: -2,
    left: -2,
    width: 40,
    height: 40,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderColor: "#4CAF50",
    borderBottomLeftRadius: 8,
  },
  cornerBottomRight: {
    position: "absolute",
    bottom: -2,
    right: -2,
    width: 40,
    height: 40,
    borderBottomWidth: 4,
    borderRightWidth: 4,
    borderColor: "#4CAF50",
    borderBottomRightRadius: 8,
  },
  guideText: {
    color: "white",
    fontSize: 16,
    textAlign: "center",
    marginTop: 20,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    padding: 10,
    borderRadius: 8,
  },
  livenessContainer: {
    position: "absolute",
    bottom: 120,
    left: 40,
    right: 40,
  },
  livenessBar: {
    height: 8,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
    borderRadius: 4,
    overflow: "hidden",
  },
  livenessProgress: {
    height: "100%",
    backgroundColor: "#4CAF50",
  },
  livenessText: {
    color: "white",
    textAlign: "center",
    marginTop: 8,
    fontSize: 14,
  },
  bottomControls: {
    position: "absolute",
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  captureButton: {
    backgroundColor: "#4CAF50",
    paddingVertical: 16,
    paddingHorizontal: 48,
    borderRadius: 30,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  captureButtonText: {
    color: "white",
    fontSize: 18,
    fontWeight: "bold",
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0, 0, 0, 0.8)",
    justifyContent: "center",
    alignItems: "center",
    gap: 20,
  },
  overlayText: {
    color: "white",
    fontSize: 18,
  },
  retryButton: {
    backgroundColor: "#2196F3",
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 20,
  },
  button: {
    backgroundColor: "#4CAF50",
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 20,
    marginTop: 20,
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "bold",
  },
});

export default DocumentCapture;
