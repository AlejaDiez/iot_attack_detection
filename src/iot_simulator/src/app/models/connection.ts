import { Device } from "@models/device";
import { Router } from "@models/router";
import { Packet } from "./packet";

/**
 * Enumeración de los estados de transmisión de la conexión.
 */
export enum TransmittingStatus {
    NONE,
    ROUTER_TO_DEVICE,
    DEVICE_TO_ROUTER,
}

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
    public transmitting: TransmittingStatus;

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
        this.transmitting = TransmittingStatus.NONE;
    }

    /**
     * Transmite un paquete desde el dispositivo al router o viceversa.
     *
     * @param packet Paquete a transmitir.
     */
    public spreadPacket(packet: Packet): void {
        if (packet.srcIP === this.device.ip)
            this.transmitting = TransmittingStatus.DEVICE_TO_ROUTER;
        else this.transmitting = TransmittingStatus.ROUTER_TO_DEVICE;
        setTimeout(() => {
            (this.transmitting === TransmittingStatus.DEVICE_TO_ROUTER
                ? this.router
                : this.device
            ).receivePacket(packet);
            this.transmitting = TransmittingStatus.NONE;
        }, this.latency);
    }

    /**
     * Convierte la conexión a un objeto plano.
     *
     * @returns Objeto plano con los datos de la conexión.
     */
    public toObject(): any {
        return this.latency;
    }
}
