import { Device } from "./device";

/** Representa un array de objetos Connection */
export type Connections = Connection[];

/**
 * Representa un objeto Connection.
 */
export interface Connection {
    /** Dispositivo de origen */
    origin: Device;
    /** Dispositivo de destino */
    destination: Device;
    /** Latencia de la conexión */
    latency: number;
}
