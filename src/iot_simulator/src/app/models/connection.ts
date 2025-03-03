import { Device } from "@models/device";
import { Router } from "@models/router";
import { Packet } from "./packet";

/**
 * Representa una conexión entre un dispositivo y un router. A esta conexión se le puede asignar una latencia y permite comunicar ambos dispositivos.
 */
export class Connection {
    /** Router */
    router: Router;
    /** Dispositivo */
    device: Device;
    /** Latencia de la conexión */
    latency: number;
    /* Indica si la conexión está siendo utilizada */
    public transmitting: boolean;

    /**
     * Crea una instancia de la clase Connection.
     *
     * @param router Router.
     * @param device Dispositivo.
     * @param latency Latencia de la conexión.
     */
    constructor(router: Router, device: Device, latency: number) {
        this.router = router;
        this.device = device;
        this.latency = latency;
        this.transmitting = false;
    }

    /**
     * Transmite un paquete desde el dispositivo al router o viceversa.
     *
     * @param packet Paquete a transmitir.
     */
    public spreadPacket(packet: Packet): void {
        this.transmitting = true;
        setTimeout(() => {
            if (packet.srcIP === this.device.ip) this.router.sendPacket(packet);
            else this.device.sendPacket(packet);
            this.transmitting = false;
        }, this.latency);
    }
}
