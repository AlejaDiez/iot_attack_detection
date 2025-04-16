import {
    ActivatedRouteSnapshot,
    RouterStateSnapshot,
    Routes,
} from "@angular/router";
import { validNodeGuard } from "@guards/valid-node.guard";
import { NodeType } from "@models/node";
import { AttackComponent } from "./attack/attack.component";
import { ConfigurationComponent } from "./configuration/configuration.component";
import { NetworkTrafficComponent } from "./network-traffic/network-traffic.component";
import { PanelNodeComponent } from "./panel-node.component";

export const routes: Routes = [
    {
        path: "",
        component: PanelNodeComponent,
        children: [
            {
                title: "Tráfico de red",
                path: "network-traffic",
                component: NetworkTrafficComponent,
                canActivate: [validNodeGuard],
            },
            {
                title: "Phantom Attacker",
                path: "attack",
                component: AttackComponent,
                canActivate: [
                    (
                        route: ActivatedRouteSnapshot,
                        state: RouterStateSnapshot,
                    ) =>
                        validNodeGuard(route, state, ...NodeType.AttackerTypes),
                ],
            },
            {
                title: "Configuración",
                path: "configuration",
                component: ConfigurationComponent,
                canActivate: [validNodeGuard],
            },
            { path: "**", redirectTo: "/" },
        ],
    },
];
