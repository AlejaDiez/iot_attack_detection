import { Node } from "@models/node";
import { Packet } from "@models/packet";
import { Router } from "@models/router";

/**
 * Enumeración de los estados de transmisión de la conexión.
 */
export enum TransmittingStatus {
    NONE,
    ROUTER_TO_NODE,
    NODE_TO_ROUTER,
}

/**
 * Representa una conexión entre un dispositivo y un router. A esta conexión se le puede asignar una latencia y permite comunicar ambos dispositivos.
 */
export class Connection {
    /** Router al que se conecta el dispositivo */
    public router: Router;
    /** Nodo para el que se establece la conexión */
    public node: Node;
    /** Latencia de la conexión */
    public latency: number;
    /* Indica si la conexión está siendo utilizada */
    public transmitting: TransmittingStatus;

    /**
     * Crea una instancia de la clase Connection.
     *
     * @param router Router al que se conecta el dispositivo.
     * @param node Nodo para el que se establece la conexión.
     * @param latency Latencia de la conexión, se genera aleatoriamente si no se especifica.
     */
    public constructor(router: Router, node: Node, latency?: number) {
        this.router = router;
        this.node = node;
        this.latency =
            latency ?? Math.floor(Math.random() * (2000 - 500 + 1)) + 500;
        this.transmitting = TransmittingStatus.NONE;
    }

    /**
     * Transmite un paquete desde el dispositivo al router o viceversa.
     *
     * @param packet Paquete a transmitir.
     */
    public spreadPacket(packet: Packet): void {
        if (packet.srcIP === this.node.ip)
            this.transmitting = TransmittingStatus.NODE_TO_ROUTER;
        else this.transmitting = TransmittingStatus.ROUTER_TO_NODE;
        setTimeout(() => {
            (this.transmitting === TransmittingStatus.NODE_TO_ROUTER
                ? this.router
                : this.node
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
