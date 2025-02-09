import { Connections } from "./connection";
import { Device, DeviceType } from "./device";
import { Position } from "./position";

/**
 * Representa un dispositivo en la red.
 */
export class Router extends Device {
    /**
     * Constructor de la clase router.
     *
     * @param ip - Dirección IP del router
     * @param name - Nombre del router
     * @param connections - Conexiones asociadas con el router
     * @param position - Posición del router en la red
     */
    public constructor(
        ip: string,
        name: string,
        connections: Connections = [],
        position: Position = { x: 0, y: 0 }
    ) {
        super(ip, name, DeviceType.ROUTER, connections, position);
    }
}
