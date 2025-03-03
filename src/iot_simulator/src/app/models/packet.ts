/**
 * Enum que representa varios protocolos de capa de transporte con sus números de protocolo.
 */
export enum TransportProtocol {
    ICMP = 1,
    TCP = 6,
    UDP = 17,
}

/**
 * Enum que representa varios protocolos de capa de aplicación con sus números de puerto estándar.
 */
export enum ApplicationProtocol {
    FTP = 20,
    SMTP = 25,
    DNS = 53,
    DHCP = 67,
    HTTP = 80,
}

/**
 * Enum que representa las banderas de control de TCP.
 */
export enum TCPFlags {
    FIN = 1,
    SYN = 2,
    RST = 4,
    PSH = 8,
    ACK = 16,
    URG = 32,
}

/**
 * Modelo que representa un paquete de red.
 */
export interface Packet {
    // Direcciones y puertos
    /* Dirección IP de origen */
    srcIP: string;
    /* Puerto de origen */
    srcPort: number;
    /* Dirección IP de destino */
    dstIP: string;
    /* Puerto de destino */
    dstPort: number;

    // Protocolos
    /* Protocolo de capa de transporte */
    transportProtocol: TransportProtocol;
    /* Protocolo de capa de aplicación */
    applicationProtocol: ApplicationProtocol;

    // Datos y tamaño
    /* Datos del paquete */
    payload?: string;
    /* Tamaño del paquete en bytes */
    totalBytes: number;
    /* Tamaño de las cabeceras */
    headerSize: number;
    /* Tamaño de la carga útil (sin cabeceras) */
    payloadSize: number;

    // TCP Flags
    /* Bandera TCP */
    tcpFlags: TCPFlags;

    // Información temporal
    /* Momento en que se generó el paquete */
    timestamp: Date;
    /* Tiempo de vida del paquete */
    ttl: number;
}
