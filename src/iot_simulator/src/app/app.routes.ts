import { Routes } from "@angular/router";
import { attackingNodeGuard } from "@guards/attacking-node.guard";
import { nodeExistsGuard } from "@guards/node-exists.guard";
import { AttackComponent } from "@routes/panel-node/attack/attack.component";
import { ConfigurationComponent } from "@routes/panel-node/configuration/configuration.component";
import { NetworkTrafficComponent } from "@routes/panel-node/network-traffic/network-traffic.component";
import { PanelNodeComponent } from "@routes/panel-node/panel-node.component";
import { BlankComponent } from "./app.component";
import { getNodeResolver } from "./resolvers/get-node.resolver";

export const routes: Routes = [
    {
        path: "",
        title: "IoT Simulator",
        component: BlankComponent,
    },
    {
        path: ":mac",
        canActivate: [nodeExistsGuard],
        resolve: { node: getNodeResolver },
        title: "Dispositivo",
        component: PanelNodeComponent,
        children: [
            {
                path: "network-traffic",
                resolve: { node: getNodeResolver },
                title: "Tráfico de red",
                component: NetworkTrafficComponent,
            },
            {
                path: "attack",
                canActivate: [attackingNodeGuard],
                resolve: { node: getNodeResolver },
                title: "Phantom Attacker",
                component: AttackComponent,
            },
            {
                path: "configuration",
                resolve: { node: getNodeResolver },
                title: "Configuración",
                component: ConfigurationComponent,
            },
            { path: "**", redirectTo: "/" },
        ],
    },
    { path: "**", redirectTo: "" },
];
