/**
 * Función para obtener un número entero aleatorio entre un rango.
 *
 * @param min Número mínimo (incluido).
 * @param max Número máximo (incluido).
 * @returns Número entero aleatorio entre min y max.
 */
export const randomInt = (min: number, max: number): number =>
    Math.floor(Math.random() * (max - min + 1)) + min;

/**
 * Función para obtener un string aleatorio de un tamaño determinado.
 *
 * @param size Tamaño del string.
 * @returns String aleatorio.
 */
export const randomString = (size: number): string => {
    const chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let result = "";

    for (let i = 0; i < size; i++)
        result += chars[randomInt(0, chars.length - 1)];
    return result;
};
